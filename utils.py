import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
from typing import Tuple
from matplotlib import pyplot as plt


def get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load FashionMNIST and return flattened numpy arrays."""
    train_ds = torchvision.datasets.FashionMNIST(
        root="./data",
        train=True,
        transform=transforms.ToTensor(),
        download=True,
    )
    test_ds = torchvision.datasets.FashionMNIST(
        root="./data",
        train=False,
        transform=transforms.ToTensor(),
        download=True,
    )

    # flatten 28x28 images to 784-dim vectors
    X_train = train_ds.data.numpy().reshape(-1, 28 * 28).astype(np.float64)
    y_train = train_ds.targets.numpy()
    X_test = test_ds.data.numpy().reshape(-1, 28 * 28).astype(np.float64)
    y_test = test_ds.targets.numpy()

    return X_train, X_test, y_train, y_test


def normalize(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zero-mean, unit-variance normalization.
    Mean and std are computed from training data only.
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0  # avoid division by zero for constant pixels

    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm, X_test_norm


def plot_metrics(metrics: list, save_dir: str = "plots") -> None:
    """Plot accuracy, precision, recall, and F1 vs number of PCA components."""
    os.makedirs(save_dir, exist_ok=True)

    k_vals = [m[0] for m in metrics]
    accuracies = [m[1] for m in metrics]
    precisions = [m[2] for m in metrics]
    recalls = [m[3] for m in metrics]
    f1_scores = [m[4] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(k_vals, accuracies, marker="o", linewidth=2, label="Accuracy", color="steelblue")
    ax.plot(k_vals, precisions, marker="s", linewidth=2, label="Precision", color="coral")
    ax.plot(k_vals, recalls, marker="^", linewidth=2, label="Recall", color="seagreen")
    ax.plot(k_vals, f1_scores, marker="D", linewidth=2, label="F1-Score", color="darkorchid")

    ax.set_xlabel("Number of PCA Components (k)", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Multi-Class SVM: Metrics vs PCA Components (FashionMNIST, 1-vs-Rest)", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xticks(k_vals)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "svm_metrics_vs_pca_components.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved -> {save_path}")


def plot_accuracy(metrics: list, save_dir: str = "plots") -> None:
    """Plot accuracy vs number of PCA components."""
    os.makedirs(save_dir, exist_ok=True)

    k_vals = [m[0] for m in metrics]
    accuracies = [m[1] for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(k_vals, accuracies, marker="o", linewidth=2, color="steelblue")
    for k, acc in zip(k_vals, accuracies):
        ax.annotate(f"{acc:.3f}", (k, acc), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)

    ax.set_xlabel("Number of PCA Components (k)", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("Multi-Class SVM: Accuracy vs PCA Components (FashionMNIST)", fontsize=13)
    ax.set_xticks(k_vals)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "svm_accuracy_vs_pca_components.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved -> {save_path}")


def plot_cnn_metrics(epoch_metrics: list, save_dir: str = "plots") -> None:
    """Plot accuracy, precision, recall, and F1 vs epoch for CNN training."""
    os.makedirs(save_dir, exist_ok=True)

    epochs = [m[0] for m in epoch_metrics]
    accuracies = [m[1] for m in epoch_metrics]
    precisions = [m[2] for m in epoch_metrics]
    recalls = [m[3] for m in epoch_metrics]
    f1_scores = [m[4] for m in epoch_metrics]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, accuracies, marker="o", linewidth=2, label="Accuracy", color="steelblue")
    ax.plot(epochs, precisions, marker="s", linewidth=2, label="Precision", color="coral")
    ax.plot(epochs, recalls, marker="^", linewidth=2, label="Recall", color="seagreen")
    ax.plot(epochs, f1_scores, marker="D", linewidth=2, label="F1-Score", color="darkorchid")

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("CNN: Metrics vs Epoch (FashionMNIST)", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "cnn_metrics_vs_epoch.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved -> {save_path}")
