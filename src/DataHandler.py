import torch
from torch_geometric.data import Data, Dataset
from src import BuildGraph, CreateSuperpixel
import os


class SuperpixelDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.img_paths = [...]
        self.labels = [...]

    def len(self):
        return len(self.img_paths)

    def get(self, idx):
        img = CreateSuperpixel.load_image(self.img_paths[idx])
        y = self.labels[idx]

        segments = CreateSuperpixel.run_slic(img, n_segments=100, compactness=20)
        x, edge_index = BuildGraph.superpixels_to_graph(img, segments)

        data = Data(
            x=torch.tensor(x),
            edge_index=torch.tensor(edge_index),
            y=torch.tensor([y], dtype=torch.long),
        )
        return data
