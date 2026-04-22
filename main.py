"""
Assignment 3 — Question 1: Multi-Class SVM on FashionMNIST
===========================================================
Run:
    python main.py --sr_no 26218

The SR number is used as the global random seed (same pattern as Assignment 2).
"""
import argparse
import os
import numpy as np

from utils import get_data, normalize, plot_metrics
from model import MultiClassSVM, PCA
from typing import Tuple, List


# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────

def get_hyperparameters() -> Tuple[float, int, List[float]]:
    """
    Return the fixed hyperparameters for the SVM experiment.

    Returns
    -------
    learning_rate : float       — SGD step size
    num_iters     : int         — SGD steps per binary SVM
    C_values      : List[float] — regularisation strengths; median = 1.0
                                  (log-spaced so small and large values are
                                  both well-represented)
    """
    learning_rate = 1e-4
    num_iters     = 500            # per binary SVM (10 SVMs total per k)
    C_values      = [0.01, 0.1, 1.0, 10.0, 100.0]   # median = 1.0
    return learning_rate, num_iters, C_values


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _print_metrics(k: int, acc: float, prec: float, rec: float, f1: float) -> None:
    print(
        f"  k={k:>3d} | "
        f"Accuracy={acc:.4f}  Precision={prec:.4f}  "
        f"Recall={rec:.4f}  F1={f1:.4f}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── CLI ──────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Assignment 3 Q1 — Multi-Class SVM on FashionMNIST"
    )
    parser.add_argument(
        "--sr_no", type=int, required=True,
        help="5-digit SR number used as the global random seed",
    )
    parser.add_argument(
        "--C", type=float, default=1.0,
        help=(
            "Regularisation constant for the k-sweep plot (default=1.0, "
            "the median of the C_values list). "
            "All five C values are printed in a summary table."
        ),
    )
    args = parser.parse_args()

    seed = args.sr_no
    C_fixed = args.C

    # ── Reproducibility ───────────────────────────────────────────────────────
    np.random.seed(seed)

    # ── Hyperparameters ───────────────────────────────────────────────────────
    learning_rate, num_iters, C_values = get_hyperparameters()
    k_values = [10, 20, 50, 100, 200]  # PCA components to sweep over

    print("=" * 60)
    print("Assignment 3 Q1 — Multi-Class SVM (1-vs-Rest)")
    print(f"  Seed         : {seed}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  SGD iters    : {num_iters}  (per binary SVM)")
    print(f"  C values     : {C_values}")
    print(f"  C for k-sweep: {C_fixed}")
    print(f"  k values     : {k_values}")
    print("=" * 60)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\n[1/4] Loading FashionMNIST …")
    X_train, X_test, y_train, y_test = get_data()
    print(f"  X_train: {X_train.shape}   y_train: {y_train.shape}")
    print(f"  X_test : {X_test.shape}    y_test : {y_test.shape}")

    # ── Normalise ─────────────────────────────────────────────────────────────
    print("\n[2/4] Normalising (train stats only — no data leakage) …")
    X_train, X_test = normalize(X_train, X_test)

    # ── PCA — compute ONCE with the largest k, then slice ────────────────────
    # Computing the 784×784 covariance matrix and its eigendecomposition once
    # is far cheaper than recomputing it for every k value in the sweep.
    max_k = max(k_values)
    print(f"\n[3/4] Computing full PCA (up to {max_k} components) …")
    pca_full = PCA(n_components=max_k)
    X_train_pca_full = pca_full.fit_transform(X_train)   # (60000, max_k)
    X_test_pca_full  = pca_full.transform(X_test)        # (10000, max_k)
    print(f"  PCA done. Projected shape: {X_train_pca_full.shape}")

    # ── Sweep over k — plot metrics vs PCA components ────────────────────────
    print(f"\n[4/4] Sweeping k with C={C_fixed} …\n")
    os.makedirs("plots", exist_ok=True)
    metrics = []

    for k in k_values:
        print(f"── k = {k} ──────────────────────────────────")

        # Slice: top-k components are just the first k columns
        # (columns are already sorted by descending eigenvalue in PCA.fit_transform)
        X_tr = X_train_pca_full[:, :k]   # (60000, k)
        X_te = X_test_pca_full[:,  :k]   # (10000, k)

        # Train MultiClassSVM (1-vs-Rest: 10 binary SVMs)
        svm = MultiClassSVM(num_classes=10)
        svm.fit(
            X_tr, y_train,
            C=C_fixed,
            learning_rate=learning_rate,
            num_iters=num_iters,
        )

        # Evaluate on test set
        acc  = svm.accuracy_score(X_te, y_test)
        prec = svm.precision_score(X_te, y_test)
        rec  = svm.recall_score(X_te, y_test)
        f1   = svm.f1_score(X_te, y_test)

        metrics.append((k, acc, prec, rec, f1))
        _print_metrics(k, acc, prec, rec, f1)
        print()

    # ── Results table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Results Summary  (C={C_fixed})")
    print(f"{'k':>5}  {'Accuracy':>10}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}")
    print("-" * 55)
    for k, acc, prec, rec, f1 in metrics:
        print(f"{k:>5}  {acc:>10.4f}  {prec:>10.4f}  {rec:>10.4f}  {f1:>10.4f}")
    print("=" * 60)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_metrics(metrics, save_dir="plots")
    print("\nDone! 🎉")


if __name__ == "__main__":
    main()
