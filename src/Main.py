from torch_geometric.loader import DataLoader
from src.DataHandler import SuperpixelDataset
from src.GNN import SuperpixelGAT
import torch.nn as nn
import torch

dataset = SuperpixelDataset("path/to/dataset")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = SuperpixelGAT(in_dim=5, num_classes=10)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(20):
    for batch in loader:
        opt.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        opt.step()
    print(f"Epoch {epoch}: loss={loss.item():.4f}")
