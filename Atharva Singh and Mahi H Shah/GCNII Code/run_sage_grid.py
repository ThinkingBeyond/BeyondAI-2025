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
        self.W_self = nn.Linear(in_ft, out_ft)
        self.W_neigh = nn.Linear(in_ft, out_ft)

    def forward(self, x, adj):
        # adj is row-normalized (Mean Aggregator)
        h_neigh = torch.spmm(adj, x)
        out_neigh = self.W_neigh(h_neigh)
        out_self = self.W_self(x)
        # SAGE concat equivalent (summing projected parts)
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
    base_dir = os.getcwd()
    feat_file = os.path.join(base_dir, 'new_data', dataset_name, 'out1_node_feature_label.txt')
    edge_file = os.path.join(base_dir, 'new_data', dataset_name, 'out1_graph_edges.txt')
    split_file = os.path.join(base_dir, 'splits', f'{dataset_name}_split_0.6_0.2_{split_idx}.npz')

    if not os.path.exists(feat_file): raise FileNotFoundError(f"Not found: {feat_file}")

    # Load Features/Labels
    features_dict, labels_dict = {}, {}
    with open(feat_file, 'r') as f:
        f.readline()
        for line in f:
            parts = line.strip().split('\t')
            nid = int(parts[0])
            features_dict[nid] = np.array(parts[1].split(','), dtype=np.float32)
            labels_dict[nid] = int(parts[2])

    # Build Graph
    G = nx.DiGraph()
    for nid in features_dict: G.add_node(nid)
    with open(edge_file, 'r') as f:
        f.readline()
        for line in f:
            parts = line.strip().split('\t')
            u, v = int(parts[0]), int(parts[1])
            if u in features_dict and v in features_dict: G.add_edge(u, v)

    nodes = sorted(G.nodes())
    adj = nx.adjacency_matrix(G, nodelist=nodes)
    features = np.array([features_dict[n] for n in nodes])
    labels = np.array([labels_dict[n] for n in nodes])

    # Normalize Adjacency (Mean Aggregator)
    rowsum = np.array(adj.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    adj = r_mat_inv.dot(adj)
    
    # Normalize Features
    rowsum_f = np.array(features.sum(1), dtype=np.float32)
    r_inv_f = np.power(rowsum_f, -1).flatten()
    r_inv_f[np.isinf(r_inv_f)] = 0.
    r_mat_inv_f = sp.diags(r_inv_f)
    features = r_mat_inv_f.dot(features)

    # Tensor conversion
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    adj_coo = adj.tocoo()
    indices = torch.from_numpy(np.vstack((adj_coo.row, adj_coo.col)).astype(np.int64))
    values = torch.from_numpy(adj_coo.data)
    shape = torch.Size(adj.shape)
    adj_tensor = torch.sparse.FloatTensor(indices, values, shape).float()

    if not os.path.exists(split_file): raise FileNotFoundError(f"Split not found: {split_file}")
    with np.load(split_file) as splits:
        train_mask = torch.BoolTensor(splits['train_mask'])
        val_mask = torch.BoolTensor(splits['val_mask'])
        test_mask = torch.BoolTensor(splits['test_mask'])

    return adj_tensor, features, labels, train_mask, val_mask, test_mask

# --- 3. Main Grid Search ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    datasets = ['texas', 'wisconsin', 'cornell']
    layers_list = [2, 4, 8, 16, 32]
    
    # Store results: results[dataset][layer] = "mean +/- std"
    final_results = {ds: {} for ds in datasets}

    print(f"Starting Grid Search on {device}...")
    
    for layer_num in layers_list:
        print(f"\n=== Evaluating with {layer_num} Layers ===")
        
        for ds_name in datasets:
            split_accs = []
            
            # Run 10 splits
            for i in range(10):
                try:
                    adj, features, labels, train_mask, val_mask, test_mask = load_webkb_data(ds_name, i)
                except Exception as e:
                    print(f"Skipping {ds_name} split {i}: {e}")
                    continue

                adj = adj.to(device)
                features = features.to(device)
                labels = labels.to(device)
                
                model = GraphSAGE(nfeat=features.shape[1],
                                  nlayers=layer_num,
                                  nhidden=64,
                                  nclass=labels.max().item() + 1,
                                  dropout=0.5).to(device)
                
                optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
                
                best_val_acc = 0.0
                best_test_acc = 0.0
                patience = 50
                bad_counter = 0
                
                for epoch in range(200):
                    model.train()
                    optimizer.zero_grad()
                    output = model(features, adj)
                    loss_train = F.nll_loss(output[train_mask], labels[train_mask])
                    loss_train.backward()
                    optimizer.step()
                    
                    model.eval()
                    with torch.no_grad():
                        output = model(features, adj)
                        preds = output.max(1)[1]
                        
                        acc_val = preds[val_mask].eq(labels[val_mask]).sum().item() / val_mask.sum().item()
                        acc_test = preds[test_mask].eq(labels[test_mask]).sum().item() / test_mask.sum().item()
                        
                        if acc_val > best_val_acc:
                            best_val_acc = acc_val
                            best_test_acc = acc_test
                            bad_counter = 0
                        else:
                            bad_counter += 1
                        
                        if bad_counter >= patience: break
                
                split_accs.append(best_test_acc)
            
            if split_accs:
                mean = np.mean(split_accs) * 100
                std = np.std(split_accs) * 100
                final_results[ds_name][layer_num] = f"{mean:.2f} ± {std:.2f}"
                print(f"Done {ds_name} (L={layer_num}): {mean:.2f}")

    # --- Print Table ---
    print("\n\n" + "="*65)
    print(f"{'Dataset':<12} | {'L=2':<12} | {'L=4':<12} | {'L=8':<12} | {'L=16':<12} | {'L=32':<12}")
    print("-" * 65)
    for ds in datasets:
        row = f"{ds:<12}"
        for l in layers_list:
            val = final_results[ds].get(l, "N/A")
            row += f" | {val:<12}"
        print(row)
    print("="*65 + "\n")

if __name__ == "__main__":
    main()
    
