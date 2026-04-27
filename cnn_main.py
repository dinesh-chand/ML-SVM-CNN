"""
CNN on FashionMNIST (Q2)

Run:
    python cnn_main.py --sr_no <your_sr_no>
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import CNN
from utils import plot_cnn_metrics


def evaluate(model, loader, device):
    """Run inference on a dataloader and return all predictions and true labels."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    model.train()
    return np.array(all_preds), np.array(all_labels)


def compute_metrics(preds, labels, num_classes=10):
    """Return accuracy, macro precision, recall, and F1 (no sklearn)."""
    acc = float(np.mean(preds == labels))
    p_list, r_list, f_list = [], [], []
    for cls in range(num_classes):
        tp = int(np.sum((preds == cls) & (labels == cls)))
        fp = int(np.sum((preds == cls) & (labels != cls)))
        fn = int(np.sum((preds != cls) & (labels == cls)))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        p_list.append(p)
        r_list.append(r)
        f_list.append(f)
    return acc, float(np.mean(p_list)), float(np.mean(r_list)), float(np.mean(f_list))


def main() -> None:
    parser = argparse.ArgumentParser(description="CNN on FashionMNIST")
    parser.add_argument("--sr_no", type=int, required=True, help="SR number used as random seed")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(args.sr_no)
    np.random.seed(args.sr_no)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # FashionMNIST mean/std computed from the training set
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.2860,), std=(0.3530,)),
    ])

    train_ds = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=transform, download=True)
    test_ds = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = CNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("=" * 60)
    print("CNN (LeNet-style) — FashionMNIST")
    print(f"  Seed      : {args.sr_no}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR        : {args.lr}")
    print("=" * 60)

    epoch_metrics = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        preds, labels_np = evaluate(model, test_loader, device)
        acc, prec, rec, f1 = compute_metrics(preds, labels_np)

        epoch_metrics.append((epoch, acc, prec, rec, f1))
        print(
            f"  Epoch {epoch:>2}/{args.epochs} | Loss={avg_loss:.4f} "
            f"Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}"
        )

    os.makedirs("plots", exist_ok=True)
    plot_cnn_metrics(epoch_metrics, save_dir="plots")

    # final results
    _, acc, prec, rec, f1 = epoch_metrics[-1]
    print("\n" + "=" * 60)
    print("Final Test Results")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print("=" * 60)
    print("\nDone!")


if __name__ == "__main__":
    main()
