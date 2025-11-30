import os
import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

# --- 1. Model Definition: GraphSAGE ---
class SAGEConv(nn.Module):
    def __init__(self, in_ft, out_ft):
        super(SAGEConv, self).__init__()
        # Linear layer for the node's own features
        self.W_self = nn.Linear(in_ft, out_ft)
        # Linear layer for the aggregated neighbor features
        self.W_neigh = nn.Linear(in_ft, out_ft)

    def forward(self, x, adj):
        # adj is row-normalized, so spmm calculates the mean of neighbors
        h_neigh = torch.spmm(adj, x)
        
        # Apply linear transformations
        out_neigh = self.W_neigh(h_neigh)
        out_self = self.W_self(x)
        
        # Aggregation: Summing the self and neighbor branches
        # (Mathematically equivalent to W * Concat(x, mean_neigh))
        return out_self + out_neigh

class GraphSAGE(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout):
        super(GraphSAGE, self).__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout
        self.nlayers = nlayers
        
        # Stacking layers
        self.convs.append(SAGEConv(nfeat, nhidden))
        for _ in range(nlayers - 2):
            self.convs.append(SAGEConv(nhidden, nhidden))
        
        if nlayers > 1:
            self.convs.append(SAGEConv(nhidden, nclass))
        else:
            self.convs = nn.ModuleList([SAGEConv(nfeat, nclass)])

    def forward(self, x, adj):
        for i, layer in enumerate(self.convs):
            if i < self.nlayers - 1:
                x = layer(x, adj)
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
            else:
                x = layer(x, adj)
        return F.log_softmax(x, dim=1)

# --- 2. Data Loading Utility ---
def load_webkb_data(dataset_name, split_idx):
    """
    Loads data for Texas, Wisconsin, or Cornell using the same logic as process.py
    """
    # Define Paths
    base_dir = os.getcwd() # Assumes running from repo root
    feat_file = os.path.join(base_dir, 'new_data', dataset_name, 'out1_node_feature_label.txt')
    edge_file = os.path.join(base_dir, 'new_data', dataset_name, 'out1_graph_edges.txt')
    split_file = os.path.join(base_dir, 'splits', f'{dataset_name}_split_0.6_0.2_{split_idx}.npz')

    if not os.path.exists(feat_file):
        raise FileNotFoundError(f"Data file not found: {feat_file}")

    # Parse Features and Labels
    features_dict = {}
    labels_dict = {}
    with open(feat_file, 'r') as f:
        f.readline() # Skip header
        for line in f:
            parts = line.strip().split('\t')
            node_id = int(parts[0])
            # Parse comma-separated features
            feats = np.array(parts[1].split(','), dtype=np.float32)
            label = int(parts[2])
            features_dict[node_id] = feats
            labels_dict[node_id] = label

    # Build Graph
    G = nx.DiGraph()
    # Add all nodes first to ensure order matches features
    for nid in features_dict:
        G.add_node(nid)
        
    with open(edge_file, 'r') as f:
        f.readline() # Skip header
        for line in f:
            parts = line.strip().split('\t')
            u, v = int(parts[0]), int(parts[1])
            if u in features_dict and v in features_dict:
                G.add_edge(u, v)

    # Sort nodes by ID (Crucial for aligning with masks)
    nodes = sorted(G.nodes())
    
    # Extract Matrices
    adj = nx.adjacency_matrix(G, nodelist=nodes)
    features = np.array([features_dict[n] for n in nodes])
    labels = np.array([labels_dict[n] for n in nodes])

    # Row-Normalize Adjacency for GraphSAGE (Mean Aggregator)
    # FIX: Ensure dtype is float before power operation
    rowsum = np.array(adj.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    adj = r_mat_inv.dot(adj)
    
    # Row-Normalize Features
    # FIX: Ensure dtype is float before power operation
    rowsum_f = np.array(features.sum(1), dtype=np.float32)
    r_inv_f = np.power(rowsum_f, -1).flatten()
    r_inv_f[np.isinf(r_inv_f)] = 0.
    r_mat_inv_f = sp.diags(r_inv_f)
    features = r_mat_inv_f.dot(features)

    # Convert to PyTorch Tensors
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    
    # Sparse Adjacency Tensor
    adj_coo = adj.tocoo()
    indices = torch.from_numpy(np.vstack((adj_coo.row, adj_coo.col)).astype(np.int64))
    values = torch.from_numpy(adj_coo.data)
    shape = torch.Size(adj.shape)
    adj_tensor = torch.sparse.FloatTensor(indices, values, shape).float()

    # Load Split Masks
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
        
    with np.load(split_file) as splits:
        train_mask = torch.BoolTensor(splits['train_mask'])
        val_mask = torch.BoolTensor(splits['val_mask'])
        test_mask = torch.BoolTensor(splits['test_mask'])

    return adj_tensor, features, labels, train_mask, val_mask, test_mask

# --- 3. Main Evaluation Loop ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden dimensions.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay.')
    parser.add_argument('--layers', type=int, default=2, help='Number of layers.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    datasets = ['texas', 'wisconsin', 'cornell']

    print(f"Running GraphSAGE (Mean Aggregation) on device: {device}")
    
    for ds_name in datasets:
        print(f"\n--- Dataset: {ds_name} ---")
        test_accuracies = []
        
        # Iterate over all 10 splits
        for i in range(10):
            try:
                adj, features, labels, train_mask, val_mask, test_mask = load_webkb_data(ds_name, i)
            except FileNotFoundError as e:
                print(e)
                continue

            # Move to device
            adj = adj.to(device)
            features = features.to(device)
            labels = labels.to(device)
            train_mask = train_mask.to(device)
            val_mask = val_mask.to(device)
            test_mask = test_mask.to(device)

            # Initialize Model
            model = GraphSAGE(nfeat=features.shape[1],
                              nlayers=args.layers,
                              nhidden=args.hidden,
                              nclass=labels.max().item() + 1,
                              dropout=args.dropout).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            # Training
            best_val_acc = 0.0
            best_test_acc = 0.0
            patience = 50
            bad_counter = 0

            for epoch in range(args.epochs):
                model.train()
                optimizer.zero_grad()
                output = model(features, adj)
                loss_train = F.nll_loss(output[train_mask], labels[train_mask])
                loss_train.backward()
                optimizer.step()

                # Validation
                model.eval()
                with torch.no_grad():
                    output = model(features, adj)
                    preds = output.max(1)[1]
                    
                    correct_val = preds[val_mask].eq(labels[val_mask]).sum().item()
                    acc_val = correct_val / val_mask.sum().item()
                    
                    correct_test = preds[test_mask].eq(labels[test_mask]).sum().item()
                    acc_test = correct_test / test_mask.sum().item()

                    if acc_val > best_val_acc:
                        best_val_acc = acc_val
                        best_test_acc = acc_test
                        bad_counter = 0
                    else:
                        bad_counter += 1
                
                if bad_counter >= patience:
                    break
            
            test_accuracies.append(best_test_acc)
            # print(f"Split {i}: Test Acc: {best_test_acc:.4f}")

        if test_accuracies:
            mean_acc = np.mean(test_accuracies) * 100
            std_acc = np.std(test_accuracies) * 100
            print(f"Result: {mean_acc:.2f} +/- {std_acc:.2f}")

if __name__ == "__main__":
    main()