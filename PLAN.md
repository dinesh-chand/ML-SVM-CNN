# Assignment 3 — Implementation Plan
## E0-270o Machine Learning | FashionMNIST | SVM + CNN

> **Strategy**: One question at a time. Q1 = SVM from scratch. Q2 = CNN in PyTorch.
> Reuse Assignment 2's data/split/normalize methodology wherever it maps cleanly.

---

## 🗂️ File Map (what lives where)

```
Assignment3/
├── utils.py        ← data loading, normalization, plotting  [IMPLEMENT]
├── model.py        ← SupportVectorModel, MultiClassSVM, PCA [IMPLEMENT]
├── main.py         ← orchestration for Q1 (SVM)            [FIX + IMPLEMENT]
├── cnn_main.py     ← orchestration for Q2 (CNN)            [NEW FILE - Q2]
├── LeNet5.py       ← reference impl (already done, MNIST)   [ADAPT for Q2]
├── PLAN.md         ← this file
└── THEORY.md       ← concept writeup with code bridges
```

---

## 🐛 Known Bugs in Starter Code (fix these first!)

| File | Bug | Fix |
|------|-----|-----|
| `main.py` | `PCA` imported from `model.py` but class doesn't exist | Implement `PCA` in `model.py` |
| `main.py` | Loop `for C_i in C` but uses undefined variable `k` for PCA | Restructure: loop over `k_values`, fix C=1 (median) for the k-sweep plot |
| `main.py` | `get_hyperparameters()` raises `NotImplementedError` | Implement it |
| `model.py` | All methods raise `NotImplementedError` | Implement all |
| `utils.py` | All functions raise `NotImplementedError` | Implement all |

---

## ✅ QUESTION 1: Multi-Class SVM (numpy only, no sklearn!)

### Phase 1.1 — Data Layer (`utils.py`)

**Reusing from Assignment 2:**
- Same train/test split philosophy (seeded, reproducible)
- Same `preprocess_data()` pattern (handle missing values via column means)
- Same `normalize()` approach (StandardScaler equivalent — zero mean, unit variance)

**New for Assignment 3:**
- Dataset = FashionMNIST (via `torchvision.datasets.FashionMNIST`)
- Images are 28×28 → flatten to 784-dim vectors for SVM
- Labels are integers 0–9 (already numeric, no mapping needed like Asst2's M/B)
- 60k train / 10k test (fixed split, no need to resplit)

**Functions to implement in `utils.py`:**
```
get_data()
  ├── Download FashionMNIST via torchvision
  ├── Extract images + labels as numpy arrays
  ├── Flatten images: (N, 28, 28) → (N, 784)
  └── Return X_train, X_test, y_train, y_test

normalize(X_train, X_test)
  ├── Compute mean and std from X_train ONLY (prevent data leakage — same as Asst2!)
  ├── Subtract mean, divide by std
  └── Return normalized X_train, X_test

plot_metrics(metrics)
  ├── metrics = list of (k, accuracy, precision, recall, f1)
  ├── Plot all 4 metrics vs k on same axes
  └── Save as 'metrics_vs_pca_components.png'
```

---

### Phase 1.2 — PCA from Scratch (`model.py`)

**Why PCA before SVM?**
FashionMNIST has 784 dimensions. SVM SGD on 784 dims × 60k samples is slow.
PCA reduces dimensions while retaining maximum variance — makes SVM tractable.

**PCA class to implement:**
```python
class PCA:
    def fit_transform(self, X)    # Center, compute covariance, eigendecomp, project
    def transform(self, X)        # Project new data using same eigenvectors
```

**Algorithm:**
1. Center: `X_centered = X - mean(X, axis=0)`
2. Covariance: `C = (1/n) * X_centered.T @ X_centered`
3. Eigendecomposition: `eigenvalues, eigenvectors = np.linalg.eigh(C)`
4. Sort eigenvectors by descending eigenvalue
5. Take top `k` eigenvectors → projection matrix `W` (784 × k)
6. `X_reduced = X_centered @ W`

**k values to test:** `[10, 20, 50, 100, 200]` — vary these and plot metrics

---

### Phase 1.3 — SVM from Scratch (`model.py`)

**`SupportVectorModel` (binary, one per class):**

```
_initialize(X):
  w = zeros(n_features)
  b = 0.0

fit(X, y, lr, num_iters, C):
  Loop num_iters times:
    1. Sample random index i (SGD = one sample at a time)
    2. Compute margin: margin = y[i] * (w @ X[i] + b)
    3. If margin >= 1:   (correctly classified, outside margin)
          grad_w = 2 * w         (only L2 regularizer)
          grad_b = 0
       Else:             (inside margin or misclassified — hinge kicks in)
          grad_w = 2 * w - C * y[i] * X[i]
          grad_b = -C * y[i]
    4. w -= lr * grad_w
    5. b -= lr * grad_b

predict(X):
  return sign(X @ w + b)   → returns +1 or -1
```

**`MultiClassSVM` (1-vs-rest wrapper):**

```
fit(X, y, **kwargs):
  For each class k in {0..9}:
    y_binary = where(y == k, +1, -1)   ← relabel for 1-vs-rest
    models[k].fit(X, y_binary, **kwargs)

predict(X):
  scores = matrix of shape (n_samples, 10)
  For each class k:
    scores[:, k] = X @ models[k].w + models[k].b   ← raw decision value
  return argmax(scores, axis=1)                     ← class with highest score

precision_score(X, y):   ← macro-averaged
recall_score(X, y):      ← macro-averaged
f1_score(X, y):          ← macro-averaged (2*P*R / (P+R))
```

**Why macro-average?** 10 balanced classes in FashionMNIST → macro = micro ≈ weighted.
Simple and interpretable. (Same reasoning as Asst2 used sklearn defaults.)

---

### Phase 1.4 — Fix & Wire `main.py`

**Current bug:** the loop iterates `for C_i in C` but uses variable `k` (undefined).
The assignment asks to plot **metrics vs. k (PCA components)** — so k is the loop variable.

**Corrected structure:**
```
get_hyperparameters() → lr=0.001, num_iters=1000, C_list=[0.01, 0.1, 1.0, 10.0, 100.0]

main():
  X_train, X_test, y_train, y_test = get_data()
  X_train, X_test = normalize(X_train, X_test)

  C = 1.0  # median value, fixed for the k-sweep
  k_values = [10, 20, 50, 100, 200]

  metrics = []
  for k in k_values:
    pca = PCA(n_components=k)
    X_tr_pca = pca.fit_transform(X_train)
    X_te_pca = pca.transform(X_test)

    svm = MultiClassSVM(num_classes=10)
    svm.fit(X_tr_pca, y_train, C=C, learning_rate=lr, num_iters=num_iters)

    accuracy  = svm.accuracy_score(X_test_pca, y_test)
    precision = svm.precision_score(X_test_pca, y_test)
    recall    = svm.recall_score(X_test_pca, y_test)
    f1        = svm.f1_score(X_test_pca, y_test)

    metrics.append((k, accuracy, precision, recall, f1))

  plot_metrics(metrics)
```

---

### Phase 1.5 — Q1 Testing Checklist

- [ ] `utils.py` `get_data()` returns shapes `(60000, 784)`, `(10000, 784)`, `(60000,)`, `(10000,)`
- [ ] `normalize()` — X_test uses X_train's mean/std (no leakage)
- [ ] `PCA.fit_transform()` returns shape `(60000, k)`
- [ ] `PCA.transform()` returns shape `(10000, k)` using same eigenvectors
- [ ] `SupportVectorModel.predict()` returns only +1 or -1
- [ ] `MultiClassSVM.predict()` returns class indices 0–9
- [ ] Accuracy on test set should be ~70–80% for k=100, C=1
- [ ] Plot saved to `plots/metrics_vs_pca_components.png`

---

## ✅ QUESTION 2: CNN (PyTorch) — Do this AFTER Q1 is verified!

> **Start this only after Q1 is complete and verified.**

### Phase 2.1 — Dataset (same `get_data()` from utils.py, but different form)
- For CNN, images stay as `(N, 1, 28, 28)` tensors — NO flattening
- Use PyTorch `DataLoader` with `batch_size=64`, `shuffle=True`
- `LeNet5.py` already shows this pattern — reuse it for FashionMNIST (change `MNIST` → `FashionMNIST`)

### Phase 2.2 — CNN Architecture (`cnn_main.py`)
- At least 2 Conv layers + MaxPool + FC layers (LeNet5.py is already a perfect template)
- Just swap dataset: `torchvision.datasets.FashionMNIST` instead of `MNIST`
- Add: track Precision, Recall, F1 per epoch (not just accuracy + loss)

### Phase 2.3 — Metrics vs PCA Components (for CNN)
- Interesting twist: apply PCA to the **flattened feature maps** from the penultimate layer
- OR apply PCA to the raw pixel input before CNN (same as SVM approach)
- Assignment likely means: apply PCA to raw input, then train CNN on reduced dims

### Phase 2.4 — `cnn_main.py` Structure
```
k_values = [10, 20, 50, 100, 200, 784]  # 784 = no PCA (baseline)
For each k:
  Apply PCA (from model.py) to flatten images
  Reshape back to (N, 1, sqrt(k)...) OR use as FC input directly
  Train CNN for fixed epochs
  Evaluate: accuracy, precision, recall, f1
  metrics.append((k, acc, prec, rec, f1))
plot_metrics(metrics)  # reuse from utils.py
```

---

## 📦 Dependency Requirements

```
numpy
torch
torchvision
tqdm
matplotlib
scipy (optional, for eigendecomp validation)
```

> No `scikit-learn` for Q1! (sklearn is banned for the SVM implementation)
> scikit-learn is fine for Q2 metrics if needed.

---

## 🔁 Reuse Audit (what carries over from Assignment 2)

| Component | Assignment 2 | Assignment 3 | Reuse? |
|-----------|-------------|-------------|--------|
| Data split | `train_test_split(70/30, seed)` | FashionMNIST fixed 60k/10k split | ❌ Different dataset |
| Normalization | StandardScaler fit on train only | Same philosophy! | ✅ Port the pattern |
| Missing value impute | `np.nanmean` fill | FashionMNIST has no missing values | ❌ Not needed |
| Metrics (accuracy/prec/recall/F1) | sklearn + custom | Custom from scratch for SVM | ✅ Concept reuse |
| Plot patterns | `matplotlib` line plots | Same | ✅ Copy the style |
| Seeded reproducibility | `seed = sr_no` | `seed = sr_no` | ✅ Same pattern |
| tqdm progress bars | ✅ Used | ✅ Use in SGD loop | ✅ |
| Argparse for sr_no | ✅ `--sr_no` | ✅ Same | ✅ |

---

## 🎯 Implementation Order

```
Day 1: Q1
  Step 1: utils.py (get_data, normalize, plot_metrics)         ~30 min
  Step 2: model.py PCA class                                   ~30 min
  Step 3: model.py SupportVectorModel                          ~45 min
  Step 4: model.py MultiClassSVM                               ~30 min
  Step 5: Fix main.py, run, verify outputs                     ~30 min

Day 2: Q2
  Step 6: cnn_main.py based on LeNet5.py + FashionMNIST        ~45 min
  Step 7: PCA+CNN integration, metrics tracking                 ~30 min
  Step 8: Final plots, report write-up                          ~60 min
```

---

## ⚠️ Gotchas to Watch Out For

1. **PCA must be fit on training data only** — never fit on test data (data leakage!)
2. **SVM labels must be +1/-1** (not 0/1) — hinge loss formula requires this
3. **SGD is stochastic** — set `np.random.seed(sr_no)` before the loop for reproducibility
4. **FashionMNIST pixel values are 0–255** — normalize BEFORE PCA for stable gradients
5. **Covariance matrix is 784×784** — eigendecomposition might be slow; use `np.linalg.eigh` (for symmetric matrices, faster than `eig`)
6. **tqdm in SGD loop** — `num_iters` should be reasonable; 500–1000 iters per binary SVM × 10 classes = 5k–10k total SGD steps
7. **Macro-average for multiclass metrics** — be explicit, avoid `average='binary'` which is sklearn's default
