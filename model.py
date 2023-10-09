import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

seed=816
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ---------------------
# GCN
# ---------------------
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 32)
        self.conv2 = GCNConv(32, 64)

    def forward(self, edge_index, x):

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index)
        z = F.leaky_relu(x)

        return z