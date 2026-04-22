import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
from typing import Tuple
from matplotlib import pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────────────

def get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Download FashionMNIST (if not cached) and return flat numpy arrays.

    FashionMNIST has a fixed 60k/10k train/test split — no need to re-split.
    Images are 28×28 grayscale; we flatten to 784-dim vectors for the SVM.

    Returns
    -------
    X_train : (60000, 784)  float64  raw pixel values [0, 255]
    X_test  : (10000, 784)  float64
    y_train : (60000,)      int64    class indices 0–9
    y_test  : (10000,)      int64
    """
    # ToTensor() scales pixels to [0, 1] — we undo that via .numpy() on .data
    # which gives the raw uint8 values. We cast to float64 ourselves so that
    # normalize() works cleanly with double precision (same as Asst2).
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

    # .data holds raw uint8 tensors (N, 28, 28) — reshape to (N, 784)
    X_train = train_ds.data.numpy().reshape(-1, 28 * 28).astype(np.float64)
    y_train = train_ds.targets.numpy()
    X_test  = test_ds.data.numpy().reshape(-1, 28 * 28).astype(np.float64)
    y_test  = test_ds.targets.numpy()

    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────────────────────────────────────
# Normalization
# ──────────────────────────────────────────────────────────────────────────────

def normalize(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zero-mean, unit-variance normalization (StandardScaler equivalent).

    Statistics are computed from X_train ONLY — never from X_test.
    This mirrors Assignment 2's pattern:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled   = scaler.transform(X_val)

    Constant pixels (std == 0, e.g. border pixels in FashionMNIST) are left
    at zero rather than producing NaN — std is clamped to 1 for those pixels.

    Returns
    -------
    X_train_norm, X_test_norm : float64 arrays, same shape as inputs
    """
    mean = X_train.mean(axis=0)           # (784,)
    std  = X_train.std(axis=0)            # (784,)
    std[std == 0] = 1.0                   # avoid div-by-zero for dead pixels

    X_train_norm = (X_train - mean) / std
    X_test_norm  = (X_test  - mean) / std

    return X_train_norm, X_test_norm


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_metrics(metrics: list, save_dir: str = "plots") -> None:
    """
    Plot Accuracy, Precision, Recall, and F1-Score vs number of PCA components
    on a single figure with four clearly distinguished lines.

    Parameters
    ----------
    metrics  : list of tuples (k, accuracy, precision, recall, f1)
               k = number of PCA components used
    save_dir : directory in which to save the figure (created if missing)
    """
    os.makedirs(save_dir, exist_ok=True)

    k_vals     = [m[0] for m in metrics]
    accuracies = [m[1] for m in metrics]
    precisions = [m[2] for m in metrics]
    recalls    = [m[3] for m in metrics]
    f1_scores  = [m[4] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(k_vals, accuracies, marker="o", linewidth=2,
            label="Accuracy",  color="steelblue")
    ax.plot(k_vals, precisions, marker="s", linewidth=2,
            label="Precision", color="coral")
    ax.plot(k_vals, recalls,    marker="^", linewidth=2,
            label="Recall",    color="seagreen")
    ax.plot(k_vals, f1_scores,  marker="D", linewidth=2,
            label="F1-Score",  color="darkorchid")

    ax.set_xlabel("Number of PCA Components (k)", fontsize=13)
    ax.set_ylabel("Score",                         fontsize=13)
    ax.set_title("Multi-Class SVM: Metrics vs PCA Components\n(FashionMNIST, 1-vs-Rest)",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xticks(k_vals)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "svm_metrics_vs_pca_components.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {save_path}")
