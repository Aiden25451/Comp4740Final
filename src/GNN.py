import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool


class SuperpixelGAT(nn.Module):
    def __init__(self, in_dim, hidden=64, num_classes=10):
        super().__init__()
        self.g1 = GATConv(in_dim, hidden, heads=4, concat=True)
        self.g2 = GATConv(hidden * 4, hidden, heads=4, concat=False)
        self.lin = nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.g1(x, edge_index).relu()
        x = self.g2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x)
