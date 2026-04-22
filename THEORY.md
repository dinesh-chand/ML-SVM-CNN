# Assignment 3 — Theory & Concept Guide
## E0-270o Machine Learning | Topics: SVM, PCA, CNN

> **Purpose:** This document explains the *why* behind every algorithm — and then
> shows exactly *where* that math lives in the code. Read theory → look at code → it clicks.

---

# PART A — QUESTION 1: Multi-Class SVM from Scratch

---

## 1. The Big Picture: What is a Support Vector Machine?

### The Core Idea — Maximum Margin Classification

Imagine you have two classes of points on a 2D plane (say, T-shirts vs Trousers).
A linear classifier draws a straight line to separate them. There are infinitely
many lines that can separate them — so which one do you pick?

**SVM's answer:** Pick the line that has the *maximum margin* — the largest gap
between the line and the nearest data points from each class.

Those nearest points are called **support vectors** — the "supports" holding the
decision boundary in place. Everything else in the dataset is irrelevant once you
find these points. This is one of SVM's elegant properties.

```
Class -1     |      Class +1
             |
   o         |              x
             |
 o     o     |         x        x
      o  <---+---> x
             |         x
         margin = 2/‖w‖
```

### Mathematical Formulation

The decision boundary is a hyperplane: `wᵀx + b = 0`

For a binary SVM with labels y ∈ {-1, +1}:
- Points with `yᵢ(wᵀxᵢ + b) ≥ 1` are correctly classified with sufficient margin
- We want to maximize margin = `2 / ‖w‖`, which is equivalent to minimizing `‖w‖²`

**Hard SVM (linearly separable data):**
```
minimize:    (1/2) ‖w‖²
subject to:  yᵢ(wᵀxᵢ + b) ≥ 1   for all i
```

But real data is NEVER perfectly separable. Enter Soft SVM.

---

## 2. Soft SVM — Tolerating Mistakes with Slack Variables

### Why "Soft"?

Hard SVM can fail entirely if data is not linearly separable. Soft SVM introduces
**slack variables** ξᵢ ≥ 0 that allow some points to violate the margin constraint.
The penalty for violating is controlled by hyperparameter **C**.

**Soft SVM Primal (as given in the assignment):**
```
minimize:    ‖w‖²  +  C · Σᵢ ξᵢ
subject to:  yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ
             ξᵢ ≥ 0
```

**C is the trade-off knob:**
- **Large C** → penalize violations heavily → narrow margin, fits training data tightly (may overfit)
- **Small C** → tolerate violations → wide margin, more generalizable (may underfit)

That's why the assignment asks you to try `C ∈ [0.01, 0.1, 1.0, 10.0, 100.0]` —
the median C=1 is a balanced default.

---

## 3. The Hinge Loss — Eliminating Slack Variables

The constrained formulation above is hard to optimize with SGD. We can fold the
constraint directly into the objective via the **hinge loss**.

**Key insight:** The slack variable ξᵢ, when optimal, equals:
```
ξᵢ = max(0, 1 - yᵢ(wᵀxᵢ + b))
```

Substituting into the objective gives the **unconstrained hinge loss formulation**:
```
minimize:  f(w, b) = ‖w‖²  +  C · Σᵢ max(0, 1 - yᵢ(wᵀxᵢ + b))
                     ───────    ─────────────────────────────────
                     L2 reg     Hinge loss (sum over all samples)
```

This is what you actually implement! The `max(0, ...)` is the "hinge" shape —
zero when the point is correctly classified with margin, otherwise growing linearly.

### 📍 Code Bridge — Where This Lives

In `model.py`, `SupportVectorModel.fit()`:

```python
# This is the hinge loss check per sample (SGD picks one sample at a time):
margin = y[i] * (self.w @ X[i] + self.b)

if margin >= 1:
    # Point is correctly classified with sufficient margin
    # Only the L2 regularizer gradient applies: d/dw (‖w‖²) = 2w
    grad_w = 2 * self.w
    grad_b = 0.0
else:
    # Point violates margin — hinge loss is active
    # d/dw (‖w‖² + C * max(0, 1 - y(wᵀx+b))) = 2w - C*y*x
    grad_w = 2 * self.w - C * y[i] * X[i]
    grad_b = -C * y[i]

self.w -= learning_rate * grad_w
self.b -= learning_rate * grad_b
```

The `if margin >= 1` is literally the hinge: if you're correctly classified with
margin, the loss is zero — the hinge "turns off". Otherwise, it's linear in the
margin violation.

---

## 4. Subgradients — Handling Non-Differentiability

The hinge function `max(0, 1 - z)` is not differentiable at `z = 1` (the kink).
Normal gradient descent breaks at non-differentiable points.

**Subgradient descent** fixes this: at any non-differentiable point, instead of the
true gradient, we use a **subgradient** — any value from the set of slopes that
"support" the function from below.

At `z = 1` (the kink): the subgradient can be anything in `[-1, 0]`.
Convention: use 0 (treat the boundary as "correctly classified").

### 📍 Code Bridge

In our SGD loop, we implicitly handle this:
- `if margin >= 1:` → we're at or past the kink → use grad = 2w (reg only, hinge=0)
- `else:` → we're inside the margin → use the hinge subgradient

The boundary case `margin == 1` is handled by the `>=` condition (uses grad=2w),
which is a valid subgradient choice.

---

## 5. Stochastic Gradient Descent (SGD) for SVM

**Full gradient descent** would compute the gradient over ALL N training samples
at each step → expensive when N=60,000 and we're doing this 10 times (1-vs-rest).

**SGD** picks ONE random sample per step. The gradient of a single sample is a
*noisy estimate* of the true gradient, but it's 60,000× cheaper per step.
With enough iterations, the noisy estimates average out and the model converges.

### 📍 Code Bridge

```python
for i in tqdm(range(1, num_iters + 1)):
    # SGD: randomly pick ONE sample
    idx = np.random.randint(0, len(X))
    xi, yi = X[idx], y[idx]
    
    # compute margin for this single sample
    margin = yi * (self.w @ xi + self.b)
    # ... gradient update as shown above
```

The `tqdm` wrapper gives you a progress bar — same pattern as Assignment 2's
training loops. Familiar territory! 🐾

---

## 6. 1-vs-Rest (OvR) — Extending to 10 Classes

SVM is naturally a **binary classifier** (two classes: +1 and -1). FashionMNIST
has 10 classes. We need a strategy to extend to multi-class.

**1-vs-Rest (OvR) strategy:**
- Train 10 separate binary SVMs, one per class
- For class `k`: relabel data as +1 if label==k, else -1
- At prediction time: run all 10 SVMs, pick class with highest raw score `wᵀx + b`
  (the score, not the sign — because different SVMs have different margins)

```
Training class 0 (T-shirt):   T-shirts=+1, everything else=-1 → train SVM₀
Training class 1 (Trouser):   Trousers=+1, everything else=-1 → train SVM₁
...
Training class 9 (Ankle boot): Boots=+1, everything else=-1 → train SVM₉

Prediction on new x:
  score₀ = w₀ᵀx + b₀    ← how "T-shirt-like" is x?
  score₁ = w₁ᵀx + b₁    ← how "Trouser-like" is x?
  ...
  predicted_class = argmax([score₀, score₁, ..., score₉])
```

### 📍 Code Bridge

In `model.py`, `MultiClassSVM.fit()`:
```python
for k in range(self.num_classes):
    # 1-vs-rest relabeling (THIS IS THE KEY STEP)
    y_binary = np.where(y == k, +1, -1)
    # Train binary SVM for class k
    self.models[k].fit(X, y_binary, **kwargs)
```

In `MultiClassSVM.predict()`:
```python
# Collect raw decision scores from all 10 SVMs
scores = np.column_stack([
    X @ self.models[k].w + self.models[k].b
    for k in range(self.num_classes)
])  # shape: (n_samples, 10)

return np.argmax(scores, axis=1)  # class with highest score wins
```

---

## 7. Evaluation Metrics — Precision, Recall, F1 (Multiclass)

### Binary case (review)
- **Accuracy** = (TP + TN) / (TP + TN + FP + FN)
- **Precision** = TP / (TP + FP)  ← "of all I said positive, how many were?"
- **Recall** = TP / (TP + FN)     ← "of all actual positives, how many did I find?"
- **F1** = 2 × P × R / (P + R)   ← harmonic mean of precision and recall

### Extending to 10 classes — Macro-average

For each class `k`, compute Pₖ and Rₖ treating class k as "positive" and all
others as "negative". Then average:

```
Macro-Precision = (1/10) Σₖ Pₖ
Macro-Recall    = (1/10) Σₖ Rₖ
Macro-F1        = (1/10) Σₖ F1ₖ    (or 2*MacroP*MacroR / (MacroP+MacroR))
```

FashionMNIST has **balanced classes** (6,000 train samples per class), so macro
= micro ≈ weighted. Macro is the simplest and most interpretable choice.

### 📍 Code Bridge

In `model.py`, `MultiClassSVM.precision_score()`:
```python
y_pred = self.predict(X)
precisions = []
for k in range(self.num_classes):
    tp = np.sum((y_pred == k) & (y == k))
    fp = np.sum((y_pred == k) & (y != k))
    precision_k = tp / (tp + fp + 1e-9)   # +epsilon avoids div by zero
    precisions.append(precision_k)
return np.mean(precisions)   # macro average
```

Same pattern for recall and F1.

---

# PART B — PCA (Principal Component Analysis)

---

## 8. Why Dimensionality Reduction Before SVM?

FashionMNIST images are 28×28 = **784 dimensions**. Each SVM binary model has
a weight vector `w` of size 784. Training 10 SVMs with SGD on 60k samples ×
784 dims is computationally expensive and SVM performance can degrade in very
high dimensions (curse of dimensionality).

**PCA** finds the directions of maximum variance in the data and projects onto
the top-k of these directions. The intuition: most of the "information" in
fashion images is in the large-scale patterns (overall shape, texture gradients),
not in individual pixel noise. PCA separates signal from noise.

---

## 9. PCA — The Algorithm

### Step 1: Center the Data
```
μ = (1/N) Σᵢ xᵢ         (mean of training data)
X_centered = X - μ       (subtract mean from each row)
```
Centering is mandatory — PCA finds directions of variance, and variance is
measured around the mean.

### Step 2: Compute the Covariance Matrix
```
C = (1/N) Xᵀ X   (where X is already centered)
```
C is a 784×784 symmetric, positive semi-definite matrix.
Each entry C[i,j] = covariance between pixel i and pixel j across all images.

### Step 3: Eigendecomposition
```
C = V Λ Vᵀ
```
- **V** = matrix of eigenvectors (columns are the "principal directions")
- **Λ** = diagonal matrix of eigenvalues (how much variance each direction captures)
- Sort eigenvectors by eigenvalue in descending order

### Step 4: Project onto Top-k Eigenvectors
```
W = V[:, :k]               (784 × k matrix — the top k eigenvectors)
X_reduced = X_centered @ W  (N × k matrix — the compressed representation)
```

### 📍 Code Bridge

In `model.py`, `PCA` class:
```python
class PCA:
    def __init__(self, n_components):
        self.k = n_components
        self.mean = None         # stored from fit, used in transform
        self.components = None   # W matrix (784 × k)

    def fit_transform(self, X):
        self.mean = X.mean(axis=0)
        X_c = X - self.mean
        # Covariance matrix (use X_c.T @ X_c / N)
        C = (X_c.T @ X_c) / len(X)
        # Eigendecomposition — eigh is for symmetric matrices (faster, stable)
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        # Sort descending (eigh returns ascending order)
        idx = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, idx[:self.k]]   # (784, k)
        return X_c @ self.components                       # (N, k)

    def transform(self, X):
        # CRITICAL: use training mean, not test mean (prevents data leakage!)
        X_c = X - self.mean
        return X_c @ self.components
```

**Why `np.linalg.eigh` and not `np.linalg.eig`?**
- `eigh` is for real symmetric/Hermitian matrices (covariance matrix is always symmetric)
- `eigh` is numerically more stable and guaranteed to return real eigenvalues
- `eig` might return tiny imaginary parts due to floating point — messy

**Why fit on train, transform both train & test?**
This is the same principle as Assignment 2's `StandardScaler.fit_transform(X_train)`
followed by `scaler.transform(X_val)`. If you fit PCA on test data too, you're
"peeking" at the test distribution during training — that's data leakage.

---

## 10. What Does "Metrics vs PCA Components" Tell You?

The assignment asks you to plot Accuracy/Precision/Recall/F1 as a function of k
(number of PCA components kept):

- **Too few components (k=10):** Compressed too aggressively → low accuracy (lost signal)
- **Sweet spot (k=50–100):** Good accuracy, fast training — the components capture
  the class-discriminative variance
- **Many components (k=200+):** Diminishing returns; may even degrade slightly
  (noisy dimensions reintroduced)

This is a classic **bias-variance tradeoff** manifested through dimensionality:
- Low k → high bias (underfit), low variance
- High k → low bias (better fit), but slower + potential overfitting

### 📍 Code Bridge (main.py loop structure)

```python
k_values = [10, 20, 50, 100, 200]
metrics = []
for k in k_values:
    pca = PCA(n_components=k)
    X_tr = pca.fit_transform(X_train)   # fit on train!
    X_te = pca.transform(X_test)        # transform test with train's PCA

    svm = MultiClassSVM(num_classes=10)
    svm.fit(X_tr, y_train, C=1.0, ...)
    
    acc  = svm.accuracy_score(X_te, y_test)
    prec = svm.precision_score(X_te, y_test)
    rec  = svm.recall_score(X_te, y_test)
    f1   = svm.f1_score(X_te, y_test)
    metrics.append((k, acc, prec, rec, f1))

# Then plot_metrics(metrics) draws all 4 curves vs k
```

---

# PART C — QUESTION 2: Convolutional Neural Networks

> **Read this section AFTER completing Q1.**

---

## 11. Why Can't We Just Use MLPs on Images?

In Assignment 2, you used MLPs on tabular data (breast cancer features). Each
feature was a meaningful scalar. For images, the raw input to an MLP would be
all 784 pixels flattened. This destroys **spatial structure**:
- Pixel (5,5) is adjacent to pixel (5,6) — they should be treated similarly
- An MLP sees them as unrelated features

Also, MLPs have no concept of **translation invariance**: a T-shirt in the top-left
vs bottom-right of the image are different inputs to an MLP, but they're the same
shirt. CNNs solve both problems.

---

## 12. The Convolution Operation — Shared Weights + Spatial Structure

A **convolutional layer** slides a small filter (kernel) over the image and computes
dot products at each position. If a filter detects "vertical edges," it fires
wherever vertical edges appear — regardless of WHERE in the image they are.

```
Input image (28×28)
      ↓
[Filter 1: edge detector] → Feature Map 1 (26×26 with 5×5 kernel, no padding)
[Filter 2: blob detector] → Feature Map 2
...
[Filter 6: ...]           → Feature Map 6
      ↓ (6 feature maps stacked)
Output: (6, 26, 26)  ← 6 channels, spatially reduced
```

**Why "shared weights"?** The same filter weights are used at every spatial
position. This massively reduces parameters compared to a fully-connected layer
scanning the same receptive field. A conv layer with 6 filters of size 5×5 has
only 6 × (5×5 + 1) = 156 parameters, vs 784 × 784 = 614,656 for an FC layer.

---

## 13. Max Pooling — Spatial Compression

After each conv layer, a **max pooling** layer (typically 2×2) takes the maximum
value in each 2×2 patch, halving the spatial dimensions:
```
(6, 26, 26) → MaxPool(2×2) → (6, 13, 13)
```

**Why max?** Max pooling retains the strongest activation in each region —
essentially answering "does this feature appear anywhere in this patch?" It:
1. Reduces computation for subsequent layers
2. Provides a form of translation invariance (small shifts don't change the max)
3. Prevents overfitting by reducing feature map size

---

## 14. Full CNN Architecture (LeNet-5 style for Assignment 3)

The provided `LeNet5.py` is already a complete implementation for MNIST.
You'll adapt it for FashionMNIST (same size images, same 10 classes):

```
Input: (batch, 1, 32, 32)   ← grayscale, resized from 28→32
    ↓
Conv(1→6, 5×5) + BatchNorm + ReLU → (batch, 6, 28, 28)
MaxPool(2×2)                       → (batch, 6, 14, 14)
    ↓
Conv(6→16, 5×5) + BatchNorm + ReLU → (batch, 16, 10, 10)
MaxPool(2×2)                        → (batch, 16, 5, 5)
    ↓
Flatten → (batch, 400)
FC(400→120) + ReLU
FC(120→84) + ReLU
FC(84→10)                           ← 10 logits (one per class)
    ↓
CrossEntropyLoss during training
Softmax → class probabilities during inference
```

---

## 15. Cross-Entropy Loss — The "Right" Loss for Classification

For multi-class classification, we don't use MSE. We use **cross-entropy loss**.

If the true class is `y` and the model outputs logits `z₀, z₁, ..., z₉`:
```
Softmax: p_k = exp(z_k) / Σⱼ exp(z_j)    ← convert logits to probabilities
Cross-entropy: L = -log(p_y)              ← penalize low probability on true class
```

Intuitively: if the model assigns p_y = 0.99 (very confident, correct), loss ≈ 0.01.
If p_y = 0.01 (very wrong), loss ≈ 4.6. PyTorch's `nn.CrossEntropyLoss` combines
softmax + cross-entropy in one numerically stable operation.

**Contrast with Assignment 2:** You used `nn.BCEWithLogitsLoss` for binary
classification. For 10-class classification, the generalization is cross-entropy.

### 📍 Code Bridge

In `LeNet5.py` (already implemented):
```python
cost = nn.CrossEntropyLoss()    # softmax + cross-entropy fused

# In training loop:
outputs = model(images)         # raw logits, shape (batch, 10)
loss = cost(outputs, labels)    # labels are class indices 0-9
```

---

## 16. Backpropagation Through Conv Layers

You don't implement this manually (PyTorch's `autograd` handles it), but it's
worth understanding conceptually.

Backprop through conv layers computes gradients w.r.t. the filter weights by
performing a **cross-correlation** of the upstream gradient with the input.
The same weight-sharing that makes forward pass efficient also makes the gradient
computation efficient — the filter gradient accumulates contributions from all
spatial positions.

### 📍 Code Bridge

```python
loss.backward()       # autograd traces the computation graph backward
                      # through FC → flatten → MaxPool → Conv → Conv
optimizer.step()      # update ALL parameters (conv filters + FC weights + biases)
optimizer.zero_grad() # reset gradients for next batch
```

This is identical to Assignment 2's training loop structure — PyTorch's autograd
abstracts away the complexity of backprop through conv layers.

---

## 17. BatchNorm — Stabilizing Deep Network Training

`nn.BatchNorm2d` normalizes the activations within each mini-batch:
- Computes mean and variance of each channel across the batch
- Normalizes: `x̂ = (x - μ) / σ`
- Applies learnable scale and shift: `y = γ x̂ + β`

**Why?** Prevents "internal covariate shift" — as weights update, the distribution
of inputs to each layer shifts, making training unstable. BatchNorm keeps the
distributions consistent, allowing higher learning rates and faster convergence.

Note: `LeNet5.py` already includes BatchNorm. The original 1998 LeNet-5 paper
didn't have it — this is a modern enhancement.

---

## 18. Adam Optimizer — Adaptive Learning Rates

In Assignment 2, you used Adam for the MLP. Assignment 3's CNN can also use Adam.

**Adam = SGD + momentum + adaptive per-parameter learning rates:**
- Maintains a running average of gradients (like momentum)
- Maintains a running average of squared gradients (scales learning rate per parameter)
- Parameters with consistently large gradients get smaller effective learning rates
  (prevents overshooting in steep directions)
- Parameters with small gradients get larger effective learning rates (explores flat regions)

For CNNs, Adam is usually preferred over vanilla SGD because different layers
(conv filters vs FC weights vs biases) have very different gradient magnitudes.

---

## 19. Comparing SVM vs CNN on FashionMNIST

| Aspect | SVM (Q1) | CNN (Q2) |
|--------|----------|----------|
| Architecture | Linear classifier (after PCA) | Hierarchical feature extractor |
| Feature engineering | PCA (manual) | Learned by conv layers (automatic) |
| Translation invariance | None (relies on PCA) | Built-in via shared weights + pooling |
| Training algorithm | SGD on hinge loss (you implement!) | Backprop via autograd |
| Expected accuracy | ~70–80% | ~88–92% |
| Interpretability | High (w = feature importance) | Lower (conv filters are visual) |
| ML library usage | numpy ONLY | PyTorch allowed |

---

## 20. The Data Pipeline — Reuse from Assignment 2

Assignment 2 taught you a clean data pipeline. Assignment 3 reuses the same
philosophy even though the dataset is different:

| Step | Assignment 2 | Assignment 3 |
|------|-------------|-------------|
| **Load** | `pd.read_csv('data.csv')` | `torchvision.datasets.FashionMNIST(download=True)` |
| **Shuffle** | `df.sample(frac=1, random_state=seed)` | FashionMNIST is pre-shuffled; use `DataLoader(shuffle=True)` |
| **Split** | `train_test_split(test_size=0.3)` | Fixed 60k/10k split (provided by dataset) |
| **Impute** | `np.nanmean` fill for NaN | Not needed (FashionMNIST has no missing pixels) |
| **Normalize** | `StandardScaler.fit(X_train).transform(both)` | `(X - mean_train) / std_train` — same logic! |
| **Label encode** | `{'M': 1, 'B': 0}` | Already integers 0–9 |
| **Reproducibility** | `seed = sr_no` | `seed = sr_no` + `torch.manual_seed(seed)` |

The key principle carried over: **fit normalizer/scaler/PCA on training data only,
then apply to test data**. This prevents data leakage and gives honest evaluation.

---

## 21. Summary: Concept → Code Map

```
THEORY                              CODE LOCATION
─────────────────────────────────── ────────────────────────────────────
Hinge loss = max(0, 1-y(wᵀx+b))  → model.py: SupportVectorModel.fit()
                                       if margin >= 1: grad = 2w
                                       else: grad = 2w - C*y*x

Subgradient at kink                → handled by `if margin >= 1` (use 0)

SGD: one sample per step           → model.py: np.random.randint → single [idx]

1-vs-rest relabeling               → model.py: np.where(y == k, +1, -1)

Argmax over 10 SVM scores          → model.py: np.argmax(scores, axis=1)

Macro-average precision            → model.py: loop over k, average Pₖ

PCA: center → covariance → eigen   → model.py: PCA.fit_transform()
PCA: fit on train, apply to test   → main.py: pca.fit_transform(X_train)
                                              pca.transform(X_test)

No data leakage (same as Asst2)    → utils.py: normalize() uses X_train stats only

Conv + MaxPool + FC                → LeNet5.py / cnn_main.py: nn.Conv2d, nn.MaxPool2d

CrossEntropyLoss (10-class BCE)    → cnn_main.py: nn.CrossEntropyLoss()

Adam optimizer                     → cnn_main.py: torch.optim.Adam(model.parameters())

Backprop through conv layers       → loss.backward() — autograd handles it

BatchNorm for training stability   → LeNet5.py: nn.BatchNorm2d(6)
```

---

*Authored with 🐾 by firangi — your code puppy — on a rainy weekend in May 2025.*
*Reuse the knowledge, don't repeat the work. DRY applies to understanding too.*
