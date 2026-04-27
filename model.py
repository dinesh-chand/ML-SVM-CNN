import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn


class PCA:
    """PCA using eigendecomposition of the covariance matrix."""

    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit PCA on training data and return the projected matrix."""
        N, d = X.shape

        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_

        cov = (X_c.T @ X_c) / N

        # eigh is for symmetric matrices and returns eigenvalues in ascending order
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        self.components_ = eigenvectors[:, :self.n_components]

        return X_c @ self.components_

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project new data using the components learned during fit_transform."""
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("Call fit_transform() before transform().")
        return (X - self.mean_) @ self.components_


class SupportVectorModel:
    """
    Binary soft-margin SVM with linear kernel, trained using SGD.
    Expects labels in {+1, -1}.
    """

    def __init__(self) -> None:
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    def _initialize(self, X: np.ndarray) -> None:
        n_features = X.shape[1]
        self.w = np.random.normal(0, 0.01, size=n_features)
        self.b = 0.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float,
        num_iters: int,
        C: float = 1.0,
    ) -> None:
        """Train using SGD with decaying learning rate (lr / (1 + t/T))."""
        self._initialize(X)
        N = len(X)

        for t in tqdm(range(num_iters), leave=False):
            i = np.random.randint(0, N)
            xi = X[i]
            yi = y[i]

            margin = yi * (self.w @ xi + self.b)

            if margin >= 1:
                grad_w = 2.0 * self.w
                grad_b = 0.0
            else:
                grad_w = 2.0 * self.w - C * yi * xi
                grad_b = -C * yi

            # linear decay: lr drops to ~lr/2 by the last step
            lr_t = learning_rate / (1 + t / num_iters)
            self.w -= lr_t * grad_w
            self.b -= lr_t * grad_b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns +1 or -1 for each sample."""
        scores = X @ self.w + self.b
        return np.where(scores >= 0, +1, -1)

    def accuracy_score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))


class MultiClassSVM:
    """
    1-vs-Rest multi-class SVM.
    Trains one binary SVM per class; prediction is argmax of raw decision scores.
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.models = [SupportVectorModel() for _ in range(num_classes)]

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train one binary SVM per class using 1-vs-Rest relabelling."""
        for cls in range(self.num_classes):
            print(f"  Training binary SVM — class {cls} vs rest ...")
            y_binary = np.where(y == cls, +1, -1)
            self.models[cls].fit(X, y_binary, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class label by taking argmax of all binary decision scores."""
        scores = np.column_stack([
            X @ self.models[cls].w + self.models[cls].b
            for cls in range(self.num_classes)
        ])
        return np.argmax(scores, axis=1)

    def accuracy_score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    def precision_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Macro-averaged precision across all classes."""
        y_pred = self.predict(X)
        per_class = []
        for cls in range(self.num_classes):
            tp = int(np.sum((y_pred == cls) & (y == cls)))
            fp = int(np.sum((y_pred == cls) & (y != cls)))
            denom = tp + fp
            per_class.append(tp / denom if denom > 0 else 0.0)
        return float(np.mean(per_class))

    def recall_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Macro-averaged recall across all classes."""
        y_pred = self.predict(X)
        per_class = []
        for cls in range(self.num_classes):
            tp = int(np.sum((y_pred == cls) & (y == cls)))
            fn = int(np.sum((y_pred != cls) & (y == cls)))
            denom = tp + fn
            per_class.append(tp / denom if denom > 0 else 0.0)
        return float(np.mean(per_class))

    def f1_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Macro-averaged F1 score across all classes."""
        y_pred = self.predict(X)
        per_class = []
        for cls in range(self.num_classes):
            tp = int(np.sum((y_pred == cls) & (y == cls)))
            fp = int(np.sum((y_pred == cls) & (y != cls)))
            fn = int(np.sum((y_pred != cls) & (y == cls)))

            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            per_class.append(f)

        return float(np.mean(per_class))


class CNN(nn.Module):
    """
    LeNet-style CNN for FashionMNIST (28x28 grayscale, 10 classes).

    Two conv blocks followed by fully connected layers.
    Input shape: (batch, 1, 28, 28)
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        # conv block 1: 1x28x28 -> 6x24x24 -> 6x12x12
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # conv block 2: 6x12x12 -> 16x8x8 -> 16x4x4
        self.block2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # 16*4*4 = 256 -> 120 -> 84 -> num_classes
        self.classifier = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = x.reshape(x.size(0), -1)
        return self.classifier(x)
