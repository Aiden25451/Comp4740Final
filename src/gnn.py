# model.py
"""
Simple GNN was heavily inspired by the PyG starting example
https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html
"""
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class SimpleGNN(nn.Module):
    def __init__(self, in_channels, hidden=64, num_classes=2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch):
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)
