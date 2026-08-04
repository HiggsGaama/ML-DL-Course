# Module 08: Handling Imbalanced Datasets

## 1. The Physical Intuition: Finding Needles in Haystacks

Imagine you are an astronomer searching for a rare type of exploding supernova star. Out of 100,000,000 telescope images taken every night, 99,999,900 show empty black space, and only 100 contain a supernova.

If you train an automated computer model by giving equal weight to every image, the model discovers a lazy shortcut: **predict "Empty Black Space" for everything!** It achieves a staggering **99.9999% accuracy**, but fails completely at its actual job.

```
Imbalanced Distribution:  [ 99.9% Normal Background Data ] vs [ 0.1% Rare Target Signals ]
Resampling & Loss Levers:  [ SMOTE Synthetic Dots / Class Weights ] ──► Rebalanced Learning Signals
```

When targets are rare (fraud detection, rare disease screening, defect detection, security intrusions), you MUST use specialized strategies to force the model to pay attention to minority signals.

---

## 2. Core Mechanics & Granular Step-by-Step Breakdown

### 1. Resampling Strategies

```
ORIGINAL DATASET:        RANDOM OVERSAMPLING:     SMOTE SYNTHETIC OVERSAMPLING:
  • • • • • • •            • • • • • • •            • • • • • • •
  • • • • • • •            • • • • • • •            • • • • • • •
  x                        x x x x x x x (Duplicates) x * x * x * x (New synthetic points)
```

1. **Random Undersampling**:
   - Randomly deletes majority class samples until majority and minority counts are equal.
   - *Trade-off*: Fast compute time, but throws away valuable majority data.

2. **Random Oversampling**:
   - Randomly duplicates existing minority class samples.
   - *Trade-off*: Preserves data size, but causes severe model overfitting onto duplicate samples.

3. **SMOTE (Synthetic Minority Over-sampling Technique)**:
   - Does NOT create duplicate copies! Instead, it looks at existing minority samples, finds their spatial neighbors, and draws synthetic new data points along the vector lines connecting those neighbors!

---

### 2. Algorithmic Loss Modifications

1. **Balanced Class Weights**:
   - Modifies the loss function penalty so misclassifying a minority sample penalizes the model $\frac{N_{\text{majority}}}{N_{\text{minority}}}$ times more heavily than misclassifying a majority sample.

2. **Focal Loss**:
   - Dynamically down-weights easy majority examples during training, forcing gradient updates to concentrate almost exclusively on hard, ambiguous minority samples.

---

## 3. Visual Architecture: SMOTE Synthetic Generation

```
Feature 2
  ^
  |        Synthetic Sample (*) generated along vector line segment
  |          o ───────── * ───────── o  (Minority Neighbor)
  |         /
  |        o (Minority Base Sample)
  |
  |    x    x     x   x (Majority Class)
  |   x   x    x   x
  +---------------------------------------------> Feature 1
```

---

## 4. Practical Implementation: SMOTE vs. Class Weights

Let's write a complete script using `imbalanced-learn` and `scikit-learn` comparing Class Weights vs. SMOTE synthetic oversampling:

```python
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, average_precision_score

# 1. Synthesize highly imbalanced dataset (95% Class 0, 5% Class 1)
X, y = make_classification(n_samples=2000, n_classes=2, weights=[0.95, 0.05], random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Original Training Distribution: 0 -> {sum(y_train==0)}, 1 -> {sum(y_train==1)}")

# --- Strategy A: Balanced Class Weights ---
weighted_rf = RandomForestClassifier(class_weight='balanced', random_state=42)
weighted_rf.fit(X_train, y_train)
y_pred_w = weighted_rf.predict(X_test)
y_prob_w = weighted_rf.predict_proba(X_test)[:, 1]

print("\n=== STRATEGY A: BALANCED CLASS WEIGHTS ===")
print(classification_report(y_test, y_pred_w))
print(f"PR-AUC (Avg Precision): {average_precision_score(y_test, y_prob_w):.4f}")

# --- Strategy B: SMOTE Synthetic Over-sampling (Fit ONLY on Training Data!) ---
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"\nSMOTE Resampled Training Distribution: 0 -> {sum(y_train_resampled==0)}, 1 -> {sum(y_train_resampled==1)}")

standard_rf = RandomForestClassifier(random_state=42)
standard_rf.fit(X_train_resampled, y_train_resampled)
y_pred_s = standard_rf.predict(X_test)
y_prob_s = standard_rf.predict_proba(X_test)[:, 1]

print("\n=== STRATEGY B: SMOTE SYNTHETIC OVERSAMPLING ===")
print(classification_report(y_test, y_pred_s))
print(f"PR-AUC (Avg Precision): {average_precision_score(y_test, y_prob_s):.4f}")
```

---

## 5. Demystifying the Mathematics: Explaining Every Formula

Let's dissect the core equations driving SMOTE and Focal Loss.

### Equation 1: SMOTE Synthetic Interpolation Formula
$$
x_{\text{new}} = x_i + \lambda (x_{zi} - x_i) \quad \text{where } \lambda \sim \text{Uniform}(0, 1)
$$

- $x_i$: A randomly selected minority class sample.
- $x_{zi}$: One of the $K$-nearest minority neighbors of $x_i$.
- $(x_{zi} - x_i)$: The difference vector pointing from $x_i$ to its neighbor.
- $\lambda$ (lambda): A random fraction between $0.0$ and $1.0$. Multiplying the difference vector by $\lambda$ places the new synthetic sample $x_{\text{new}}$ somewhere along the line connecting $x_i$ and $x_{zi}$!

---

### Equation 2: Focal Loss Formula
$$
\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

- $p_t$: The model's estimated probability for the correct ground-truth class.
- $(1 - p_t)^\gamma$: The **Modulating Factor**.
  - If a sample is easy (e.g., $p_t = 0.99$), $(1 - 0.99)^\gamma = 0.01^\gamma \approx 0$. Its loss contribution collapses to zero!
  - If a sample is hard (e.g., $p_t = 0.1$), $(1 - 0.1)^\gamma = 0.9^\gamma \approx 1$. Its loss contribution remains large!
- $\gamma$ (gamma): Tunes the down-weighting rate for easy examples (typically $\gamma = 2.0$).

---

## 6. Real-World Production Gotchas & Failure Modes

### CRITICAL MISTAKE: SMOTE Before Data Splitting
Applying SMOTE across the full dataset *before* calling `train_test_split` creates synthetic points derived from testing set samples, leaking test set distributions directly into `X_train`.
- *Fix*: Fit SMOTE strictly inside `X_train` or use `imblearn.pipeline.Pipeline`.

---

## 7. Feynman Exercises & Deep Thinking Challenges

1. **Thought Experiment**: Why does SMOTE struggle on high-dimensional data (e.g., 1,000 features)?
   - *Answer/Explanation*: Due to the **Curse of Dimensionality**, high-dimensional spaces are extremely sparse. Nearest neighbors in 1,000 dimensions are very far apart, causing SMOTE to interpolate synthetic points across empty feature space where real data never exists!

2. **Code Challenge**: Implement `Borderline-SMOTE` using `imblearn.over_sampling.BorderlineSMOTE` and compare performance against standard SMOTE.
