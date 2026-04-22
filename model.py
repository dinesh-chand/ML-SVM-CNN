import numpy as np
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# PCA — from scratch, numpy only
# ──────────────────────────────────────────────────────────────────────────────

class PCA:
    """
    Principal Component Analysis via eigendecomposition of the covariance matrix.

    Fit on training data ONLY to prevent data leakage (mirrors the StandardScaler
    pattern from Assignment 2).  Transform re-uses the training mean and the
    learned eigenvectors — never re-fits on test data.

    Algorithm
    ---------
    1. Center: X_c = X - μ_train
    2. Covariance: C = (1/N) Xᵀ X  (shape: d×d, symmetric → use eigh)
    3. Eigendecomp + sort descending by eigenvalue
    4. Project: X_reduced = X_c @ W   where W = top-k eigenvectors (d×k)
    """

    def __init__(self, n_components: int) -> None:
        self.n_components  = n_components
        self.mean_         = None   # (d,)   — set in fit_transform
        self.components_   = None   # (d, k) — top-k eigenvectors (projection matrix)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA on X (training data) and return the projected matrix.

        Parameters
        ----------
        X : (N, d) array — training data, already normalized

        Returns
        -------
        X_reduced : (N, k) array
        """
        N, d = X.shape

        # Step 1 — center
        self.mean_ = X.mean(axis=0)             # (d,)
        X_c = X - self.mean_

        # Step 2 — covariance matrix (d×d)
        # Using (1/N) Xᵀ X avoids an explicit loop; BLAS handles this efficiently.
        cov = (X_c.T @ X_c) / N                # (d, d)

        # Step 3 — eigendecomposition
        # np.linalg.eigh: for real symmetric matrices — guaranteed real eigenvalues,
        # numerically more stable than eig, returns eigenvalues in ASCENDING order.
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort descending (largest eigenvalue = most variance = most important)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]     # (d, d)

        # Keep only the top-k eigenvectors
        self.components_ = eigenvectors[:, : self.n_components]  # (d, k)

        # Step 4 — project
        return X_c @ self.components_           # (N, k)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project new data using the mean and eigenvectors learned during fit_transform.

        IMPORTANT: uses self.mean_ (training mean), NOT X's own mean.

        Parameters
        ----------
        X : (M, d) array — test / validation data

        Returns
        -------
        X_reduced : (M, k) array
        """
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("Call fit_transform() before transform().")
        return (X - self.mean_) @ self.components_  # (M, k)


# ──────────────────────────────────────────────────────────────────────────────
# Binary SVM — soft-margin, linear kernel, SGD + hinge loss
# ──────────────────────────────────────────────────────────────────────────────

class SupportVectorModel:
    """
    Binary soft-margin SVM trained with Stochastic Gradient Descent.

    Labels must be +1 / -1 (NOT 0/1).

    Objective (unconstrained form — eliminates slack variables):
        f(w, b) = ‖w‖²  +  C · Σᵢ max(0, 1 − yᵢ(wᵀxᵢ + b))
                  ──────    ────────────────────────────────
                  L2 reg     hinge loss over training samples

    SGD subgradients per sample i (sampled uniformly at random):
        If yᵢ(wᵀxᵢ + b) ≥ 1  →  ∂f/∂w = 2w,     ∂f/∂b = 0
        Else                   →  ∂f/∂w = 2w − C·yᵢ·xᵢ,  ∂f/∂b = −C·yᵢ
    """

    def __init__(self) -> None:
        self.w: np.ndarray | None = None
        self.b: float             = 0.0

    def _initialize(self, X: np.ndarray) -> None:
        """Small random initialisation — avoids all-zero symmetry breaking issues."""
        n_features = X.shape[1]
        rng = np.random.default_rng(seed=0)   # fixed seed; outer seed controls data shuffle
        self.w = rng.normal(0, 0.01, size=n_features)
        self.b = 0.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float,
        num_iters: int,
        C: float = 1.0,
    ) -> None:
        """
        Train the binary SVM via SGD.

        Parameters
        ----------
        X            : (N, k) normalized + PCA-reduced training features
        y            : (N,)   binary labels in {+1, -1}
        learning_rate: step size for gradient update
        num_iters    : total number of SGD steps (one random sample per step)
        C            : regularization trade-off (larger C = harder margin)
        """
        self._initialize(X)
        N = len(X)

        for _ in tqdm(range(num_iters), leave=False):
            # ── SGD: pick one random training sample ──────────────────────────
            i  = np.random.randint(0, N)
            xi = X[i]
            yi = y[i]

            # ── Compute functional margin ─────────────────────────────────────
            # margin = yᵢ · (wᵀxᵢ + b)
            # ≥ 1 → correctly classified with sufficient margin → hinge is zero
            # < 1 → inside or wrong side of margin → hinge is active
            margin = yi * (self.w @ xi + self.b)

            # ── Subgradient of f(w, b) w.r.t. the sampled point ──────────────
            if margin >= 1:
                # Only the L2 regularizer contributes
                # d/dw ‖w‖² = 2w
                grad_w = 2.0 * self.w
                grad_b = 0.0
            else:
                # Hinge is active: d/dw [‖w‖² + C·max(0, 1−y(wᵀx+b))]
                #   = 2w − C·y·x
                grad_w = 2.0 * self.w - C * yi * xi
                grad_b = -C * yi

            # ── Gradient descent update ───────────────────────────────────────
            self.w -= learning_rate * grad_w
            self.b -= learning_rate * grad_b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels for X.

        Returns
        -------
        y_pred : (N,) array of +1 / -1
        """
        scores = X @ self.w + self.b            # raw decision values (N,)
        return np.where(scores >= 0, +1, -1)    # sign rule

    def accuracy_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Fraction of correctly classified samples (for debugging)."""
        return float(np.mean(self.predict(X) == y))


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Class SVM — 1-vs-Rest wrapper over 10 binary SVMs
# ──────────────────────────────────────────────────────────────────────────────

class MultiClassSVM:
    """
    1-vs-Rest multi-class classifier for 10 FashionMNIST classes.

    Training:  For each class k, relabel the dataset as +1 (class k) vs -1
               (all other classes) and train a binary SupportVectorModel.

    Inference: Collect raw decision scores wₖᵀx + bₖ from all 10 models.
               Predict the class whose score is highest.

    Metrics:   Macro-averaged Precision, Recall, F1 — appropriate for the
               10 balanced FashionMNIST classes.
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.models = [SupportVectorModel() for _ in range(num_classes)]

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Train one binary SVM per class using 1-vs-Rest relabelling.

        Parameters
        ----------
        X       : (N, k) PCA-reduced, normalized training features
        y       : (N,)   multi-class labels 0–9
        **kwargs: passed directly to SupportVectorModel.fit()
                  (learning_rate, num_iters, C)
        """
        for cls in range(self.num_classes):
            print(f"  Training binary SVM — class {cls} vs rest …")

            # 1-vs-Rest relabelling:
            #   +1 → this class, -1 → everything else
            y_binary = np.where(y == cls, +1, -1)
            self.models[cls].fit(X, y_binary, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for X via argmax of raw decision scores.

        Returns
        -------
        y_pred : (N,) class indices 0–9
        """
        # Collect raw score wₖᵀx + bₖ for each of the 10 binary SVMs
        # scores[i, k] = how "class-k-like" is sample i according to SVM_k
        scores = np.column_stack([
            X @ self.models[cls].w + self.models[cls].b
            for cls in range(self.num_classes)
        ])                                          # (N, 10)

        return np.argmax(scores, axis=1)            # (N,)

    def accuracy_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Fraction of samples where predicted class matches true class."""
        return float(np.mean(self.predict(X) == y))

    # ── Macro-averaged Precision ──────────────────────────────────────────────

    def precision_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Macro-averaged precision across all 10 classes.

        For each class k:
            Precisionₖ = TP_k / (TP_k + FP_k)
            TP_k = predicted k AND actually k
            FP_k = predicted k BUT actually something else

        Macro = simple average: (1/10) Σₖ Precisionₖ
        """
        y_pred = self.predict(X)
        per_class = []
        for cls in range(self.num_classes):
            tp = int(np.sum((y_pred == cls) & (y == cls)))
            fp = int(np.sum((y_pred == cls) & (y != cls)))
            denom = tp + fp
            per_class.append(tp / denom if denom > 0 else 0.0)
        return float(np.mean(per_class))

    # ── Macro-averaged Recall ─────────────────────────────────────────────────

    def recall_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Macro-averaged recall across all 10 classes.

        For each class k:
            Recallₖ = TP_k / (TP_k + FN_k)
            FN_k = actually k BUT predicted something else
        """
        y_pred = self.predict(X)
        per_class = []
        for cls in range(self.num_classes):
            tp = int(np.sum((y_pred == cls) & (y == cls)))
            fn = int(np.sum((y_pred != cls) & (y == cls)))
            denom = tp + fn
            per_class.append(tp / denom if denom > 0 else 0.0)
        return float(np.mean(per_class))

    # ── Macro-averaged F1 ─────────────────────────────────────────────────────

    def f1_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Macro-averaged F1-score across all 10 classes.

        For each class k:
            F1ₖ = 2 · Pₖ · Rₖ / (Pₖ + Rₖ)

        Computed per-class to avoid the macro-P and macro-R harmonic mean
        shortcut (which is NOT the same as averaging per-class F1 scores).
        """
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
