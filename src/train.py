import numpy as np
import torch
import torch.nn as nn
from functools import partial
from datasets import load_dataset
from torch_geometric.loader import DataLoader

from gnn import SimpleGNN
from graph_convert import superpixels_to_graph, graph_to_data
from process_dataset import tokenize, run_slic


def build_pyg_dataset(hf_dataset):
    """Convert a HuggingFace dataset split into a list of PyG Data objects."""
    data_list = []
    for sample in hf_dataset:
        img = np.array(sample["image"].convert("RGB"))
        seg = np.array(sample["slic"])
        label = sample["label"]
        node_features, edge_index = superpixels_to_graph(img, seg)
        data = graph_to_data(node_features, edge_index, label)
        data_list.append(data)
    return data_list


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.size(0)
    return correct / total


def main():
    # ── Config ────────────────────────────────────────────────────────────
    TRAIN_SIZE = 200
    TEST_SIZE = 50
    SEGMENTS = 200
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── 1. Load & SLIC ────────────────────────────────────────────────────
    print("Loading dataset...")
    dataset = load_dataset("Simezu/brain-tumour-MRI-scan", split="train")
    dataset = dataset.train_test_split(
        test_size=TEST_SIZE, train_size=TRAIN_SIZE, seed=1
    )
    dataset = dataset.map(tokenize, batched=True, batch_size=64, num_proc=8)

    # ── 2. Build PyG datasets ─────────────────────────────────────────────
    print("Converting to graphs...")
    train_data = build_pyg_dataset(dataset["train"])
    test_data = build_pyg_dataset(dataset["test"])
    print(f"  Train graphs: {len(train_data)}, Test graphs: {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # ── 3. Model ──────────────────────────────────────────────────────────
    model = SimpleGNN(in_channels=5, hidden=64, num_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # ── 4. Train loop ─────────────────────────────────────────────────────
    print("\nEpoch | Loss   | Test Acc")
    print("------+--------+---------")
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc = evaluate(model, test_loader, device)
        print(f"  {epoch:3d} | {loss:.4f} | {acc:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
