"""
Multi-Class SVM on FashionMNIST (Q1)

Run:
    python main.py --sr_no <your_sr_no>
"""
import argparse
import os
import numpy as np

from utils import get_data, normalize, plot_metrics, plot_accuracy
from model import MultiClassSVM, PCA
from typing import Tuple, List


def get_hyperparameters() -> Tuple[float, int, List[float]]:
    """Returns learning rate, number of SGD iterations, and C values to try."""
    learning_rate = 1e-3
    num_iters = 10000
    C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    return learning_rate, num_iters, C_values


def _print_metrics(k: int, acc: float, prec: float, rec: float, f1: float) -> None:
    print(
        f"  k={k:>3d} | "
        f"Accuracy={acc:.4f}  Precision={prec:.4f}  "
        f"Recall={rec:.4f}  F1={f1:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Class SVM on FashionMNIST")
    parser.add_argument("--sr_no", type=int, required=True, help="SR number used as random seed")
    parser.add_argument("--C", type=float, default=1.0, help="Regularisation constant (default=1.0)")
    args = parser.parse_args()

    seed = args.sr_no
    C_fixed = args.C

    np.random.seed(seed)

    learning_rate, num_iters, C_values = get_hyperparameters()
    k_values = [10, 20, 50, 100, 200, 300, 400]

    print("=" * 60)
    print("Multi-Class SVM (1-vs-Rest) — FashionMNIST")
    print(f"  Seed         : {seed}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Iterations   : {num_iters}")
    print(f"  C values     : {C_values}")
    print(f"  C for k-sweep: {C_fixed}")
    print(f"  k values     : {k_values}")
    print("=" * 60)

    print("\n[1/4] Loading FashionMNIST ...")
    X_train, X_test, y_train, y_test = get_data()
    print(f"  X_train: {X_train.shape}   y_train: {y_train.shape}")
    print(f"  X_test : {X_test.shape}    y_test : {y_test.shape}")

    print("\n[2/4] Normalizing ...")
    X_train, X_test = normalize(X_train, X_test)

    # fit PCA once with the max k and slice columns for smaller k values
    max_k = max(k_values)
    print(f"\n[3/4] Computing PCA (up to {max_k} components) ...")
    pca_full = PCA(n_components=max_k)
    X_train_pca_full = pca_full.fit_transform(X_train)
    X_test_pca_full = pca_full.transform(X_test)

    print(f"\n[4/4] Sweeping k with C={C_fixed} ...\n")
    os.makedirs("plots", exist_ok=True)
    metrics = []

    for k in k_values:
        print(f"--- k = {k} ---")

        X_tr = X_train_pca_full[:, :k]
        X_te = X_test_pca_full[:, :k]

        svm = MultiClassSVM(num_classes=10)
        svm.fit(X_tr, y_train, C=C_fixed, learning_rate=learning_rate, num_iters=num_iters)

        acc = svm.accuracy_score(X_te, y_test)
        prec = svm.precision_score(X_te, y_test)
        rec = svm.recall_score(X_te, y_test)
        f1 = svm.f1_score(X_te, y_test)

        metrics.append((k, acc, prec, rec, f1))
        _print_metrics(k, acc, prec, rec, f1)
        print()

    print("\n" + "=" * 60)
    print(f"Results (C={C_fixed})")
    print(f"{'k':>5}  {'Accuracy':>10}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}")
    print("-" * 55)
    for k, acc, prec, rec, f1 in metrics:
        print(f"{k:>5}  {acc:>10.4f}  {prec:>10.4f}  {rec:>10.4f}  {f1:>10.4f}")
    print("=" * 60)

    plot_metrics(metrics, save_dir="plots")
    plot_accuracy(metrics, save_dir="plots")
    print("\nDone!")


if __name__ == "__main__":
    main()
