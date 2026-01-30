import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from process import full_load_data
from model import GraphSAGE

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay.')
parser.add_argument('--layer', type=int, default=2, help='Number of layers (GraphSAGE is usually shallow).')
parser.add_argument('--hidden', type=int, default=64, help='Hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping')
parser.add_argument('--dev', type=int, default=0, help='device id')
args = parser.parse_args()

# Setup Device
cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid if torch.cuda.is_available() else "cpu")

def train_eval(dataset_name):
    print(f"\n--- Evaluating GraphSAGE on {dataset_name} ---")
    test_accs = []
    
    # Iterate through all 10 splits provided in the 'splits' folder
    for i in range(10):
        # Path matches the structure in your uploaded files
        split_path = f"splits/{dataset_name}_split_0.6_0.2_{i}.npz"
        
        try:
            # Load data using the function from process.py
            adj, features, labels, idx_train, idx_val, idx_test, num_features, num_labels = full_load_data(dataset_name, split_path)
        except FileNotFoundError:
            print(f"Split file not found: {split_path}. Skipping.")
            continue

        # Move to device
        features = features.to(device)
        adj = adj.to(device)
        labels = labels.to(device)
        idx_train = idx_train.to(device)
        idx_val = idx_val.to(device)
        idx_test = idx_test.to(device)
        
        # Initialize Model
        model = GraphSAGE(nfeat=num_features,
                          nlayers=args.layer,
                          nhidden=args.hidden,
                          nclass=num_labels,
                          dropout=args.dropout).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        
        # Training Loop
        best_val_acc = 0.0
        best_test_acc = 0.0
        bad_counter = 0
        
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            output = model(features, adj)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                output = model(features, adj)
                # Helper for accuracy
                preds = output.max(1)[1].type_as(labels)
                correct = preds.eq(labels).double()
                
                acc_val = (correct[idx_val].sum() / len(idx_val)).item()
                acc_test = (correct[idx_test].sum() / len(idx_test)).item()
                
                if acc_val > best_val_acc:
                    best_val_acc = acc_val
                    best_test_acc = acc_test
                    bad_counter = 0
                else:
                    bad_counter += 1
                
                if bad_counter >= args.patience:
                    break
        
        test_accs.append(best_test_acc)
        print(f"Split {i}: Test Acc: {best_test_acc:.4f}")

    if test_accs:
        mean_acc = np.mean(test_accs) * 100
        std_acc = np.std(test_accs) * 100
        print(f"Result for {dataset_name}: {mean_acc:.2f} +/- {std_acc:.2f}")
    else:
        print(f"No splits found for {dataset_name}.")

# Run on all 3 WebKB datasets
datasets = ['texas', 'wisconsin', 'cornell']
for ds in datasets:
    train_eval(ds)
