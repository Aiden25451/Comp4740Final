import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch_geometric.loader import DataLoader

from gnn import SimpleGNN
from graph_convert import graph_to_data, tokenize_graph
from process_dataset import tokenize
from config import config


def build_pyg_dataset(hf_dataset):
    data_list = []
    for sample in hf_dataset:
        nf = np.array(sample["node_features"])
        ei = np.array(sample["edge_index"])
        data_list.append(graph_to_data(nf, ei, sample["label"]))
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataset = load_dataset(config["dataset_name"], split="train")
    dataset = dataset.train_test_split(
        test_size=config["test_size"],
        train_size=config["train_size"],
        seed=1,
    )

    # Superpixels
    dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=32,
        num_proc=config["num_proc"],
        load_from_cache_file=False,
    )

    # Graph conversion
    dataset = dataset.map(
        tokenize_graph,
        batched=True,
        batch_size=32,
        num_proc=config["num_proc"],
        load_from_cache_file=False,
    )

    # Convert to PyG
    train_data = build_pyg_dataset(dataset["train"])
    test_data = build_pyg_dataset(dataset["test"])

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=False)

    # ── 3. Model ──────────────────────────────────────────────────────────
    model = SimpleGNN(in_channels=15, hidden=64, num_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    print("\nEpoch | Loss   | Test Acc")
    print("------+--------+---------")

    for epoch in range(1, config["epochs"] + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc = evaluate(model, test_loader, device)
        print(f"  {epoch:3d} | {loss:.4f} | {acc:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
