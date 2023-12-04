import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

seed=816
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ---------------------
# Cross Attention
# ---------------------
class cross_attention(torch.nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        
        self.output_dim = output_dim
        self.q = nn.Parameter(torch.randn(output_dim, output_dim))
        self.v = nn.Parameter(torch.randn(output_dim, output_dim))
        self.k = nn.Parameter(torch.randn(output_dim, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.output_dim)
        for w in self.parameters():
            w.data.uniform_(-std, std)
            
    def forward(self, x1, x2):

        Q = torch.matmul(x1, self.q)
        V = torch.matmul(x1, self.v)
        K = torch.matmul(x2, self.k)
        e = F.leaky_relu(torch.matmul(Q, K.T))
        attention = F.softmax(e,dim=-1)
        x1_out = torch.matmul(attention, V)
        
        return x1_out

# ---------------------
# GCN
# ---------------------
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
    
# ---------------------
# GAT
# ---------------------
class GAT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim,4,dropout=0.2)
        self.conv2 = GATConv(hidden_dim*4, output_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x
    
# ---------------------
# GraphSAGE
# ---------------------    
class GraphSAGE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        
        return x
    
    
# ---------------------
# GraphConflator
# ---------------------
class GraphConflator(torch.nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 output_dim,
                 model):
        super().__init__()
        
        if model == 'gcn':
            self.conv1 = GCN(input_dim, hidden_dim, output_dim)
            self.conv2 = GCN(input_dim, hidden_dim, output_dim)        
        elif model == 'gat':
            self.conv1 = GAT(input_dim, hidden_dim, output_dim)
            self.conv2 = GAT(input_dim, hidden_dim, output_dim)        
        else:
            self.conv1 = GraphSAGE(input_dim, hidden_dim, output_dim)
            self.conv2 = GraphSAGE(input_dim, hidden_dim, output_dim)                 
        
        self.cross1 = cross_attention(output_dim)
        self.cross2 = cross_attention(output_dim)        

    def forward(self, 
                graph_set1, 
                graph_set2, 
                x_set1, 
                x_set2):

        out_set1 = self.conv1(x_set1, graph_set1)
        out_set2 = self.conv2(x_set2, graph_set2)
        
        # 1. 
        #logits = torch.matmul(out_set1, out_set2.T).flatten()
        
        # 2
        #logits = 0.25*(
        #    torch.matmul(out_set1, out_set1.T)+\
        #    torch.matmul(out_set1, out_set1.T)+\
        #    torch.matmul(out_set1, out_set1.T)+\
        #    torch.matmul(out_set1, out_set1.T)
        #).flatten()
        
        
        # 3. 
        out_set1 = self.cross1(out_set1, out_set2)
        out_set2 = self.cross2(out_set2, out_set1)        
        logits = torch.matmul(out_set1, out_set2.T).flatten()        
        
        return logits      