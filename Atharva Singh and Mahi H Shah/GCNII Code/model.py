import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output

class GCNII(nn.Module):
    def __init__(self, nfeat, nlayers,nhidden, nclass, dropout, lamda, alpha, variant):
        super(GCNII, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1)

class GCNIIppi(nn.Module):
    def __init__(self, nfeat, nlayers,nhidden, nclass, dropout, lamda, alpha,variant):
        super(GCNIIppi, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant,residual=True))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.sig(self.fcs[-1](layer_inner))
        return layer_inner




class GraphSAGE(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout):
        super(GraphSAGE, self).__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout
        self.nlayers = nlayers

        # Input layer
        self.convs.append(SAGEConv(nfeat, nhidden))
        
        # Hidden layers
        for _ in range(nlayers - 2):
            self.convs.append(SAGEConv(nhidden, nhidden))
            
        # Output layer (if more than 1 layer)
        if nlayers > 1:
            self.convs.append(SAGEConv(nhidden, nclass))
        else:
            # Fallback if only 1 layer is requested (rare)
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

class SAGEConv(nn.Module):
    def __init__(self, in_ft, out_ft):
        super(SAGEConv, self).__init__()
        # Weight for the node's own features
        self.W_self = nn.Linear(in_ft, out_ft)
        # Weight for the mean of neighbors
        self.W_neigh = nn.Linear(in_ft, out_ft)

    def forward(self, x, adj):
        # 1. Aggregate neighbors (Mean aggregation)
        # Note: adj is assumed to be sparse. spmm does matrix multiplication.
        # If adj is row-normalized, spmm calculates the mean.
        h_neigh = torch.spmm(adj, x)
        
        # 2. Apply linear transformations
        out_neigh = self.W_neigh(h_neigh)
        out_self = self.W_self(x)
        
        # 3. Sum parts (Equivalent to concatenation followed by a larger linear layer)
        # Standard PyTorch Geometric SAGEConv often sums these two branches 
        # (which is mathematically equivalent to W * [x || neigh])
        return out_self + out_neigh

if __name__ == '__main__':
    pass






