# Module 09: Feature Engineering & Selection

## 1. The Physical Intuition: Unlocking Hidden Signals and Cleaning Garbage

Imagine you are trying to predict whether a rocket will successfully reach orbit based on two raw telemetry logs: `Fuel Tank Mass (kg)` and `Total Vehicle Mass (kg)`. 

If you feed these two numbers separately into a linear model, the model struggles. Why? Because the physics of rocket propulsion doesn't care about fuel mass or vehicle mass individually—it cares about the **Mass Ratio**:

$$\text{Mass Ratio} = \frac{\text{Fuel Mass}}{\text{Total Mass}}$$

By calculating this ratio yourself and passing it into the model as a new feature, you have performed **Feature Engineering**. You didn't collect new data; you transformed existing data to expose the true physical law to the computer!

```
Raw Database Schema ──► [ Feature Engineering ] ──► Expanded Features (500 columns)
                              │
                              ▼
                        [ Feature Selection ] ──► Optimal Subset (20 columns) ──► Model
```

- **Feature Engineering**: Creating informative new features (ratios, interaction products, temporal cyclical transformations) that make signals obvious to models.
- **Feature Selection**: Eliminating noisy, redundant, or zero-information columns (garbage collection) to prevent overfitting and speed up execution.

---

## 2. Core Concepts & Granular Step-by-Step Breakdown

### 1. Feature Engineering Techniques

1. **Interaction & Polynomial Features**: Multiplicative combinations ($x_1 \cdot x_2$) or ratios ($\frac{x_1}{x_2}$).
2. **Cyclical Sin/Cos Temporal Transformations**: 
   - Timestamps (Hour of day: $0 \dots 23$) exhibit circular continuity: Hour 23 is a direct neighbor to Hour 0. Raw numbers $23$ and $0$ look far apart mathematically!
   - *The Solution*: Map the 24-hour clock onto a 2D circle using Sine and Cosine transformations:

```
Hour 23 (23:00) ──┐
                  ├──► Neighboring spatial points on 2D Sin/Cos Circle!
Hour 0  (00:00) ──┘
```

---

### 2. Feature Selection Methodologies

```
FILTER METHODS:       WRAPPER METHODS:             EMBEDDED METHODS:
Statistical scoring   Train model repeatedly       Selection occurs natively
(Fast, O(d))          Add/Remove features (RFE)   during training (L1 Lasso)
```

1. **Filter Methods**: Ranks features independently of model choice using statistical properties (Mutual Information, Chi-Square, Variance Threshold). Fast ($O(d)$).
2. **Wrapper Methods (RFE)**: Uses an actual ML model to evaluate candidate feature subsets recursively (Recursive Feature Elimination - RFE). Accurate, but computationally expensive ($O(d^2)$).
3. **Embedded Methods (L1 Lasso)**: Feature selection occurs natively during training. L1 Lasso regularization forces non-essential feature weights to exact zero!

---

## 3. Architecture & Visual Diagrams

### Feature Selection Approaches Comparison

```
Filter Methods:   Features ──► [ Statistical Score ] ──► Top K Features ──► Model Training
Wrapper Methods:  Features ──► [ Model Train/Eval Loop ] ──► Add/Remove ──► Optimal Subset
Embedded Methods: Features ──► [ Model Training (L1 / GBDT) ] ──► Sparsity Weights / Split Gains
```

---

## 4. Practical Implementation: Feature Selection Pipeline

Let's write a complete script using `scikit-learn` demonstrating Filter, Wrapper (RFE), and Embedded (Lasso) feature selection methods:

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression, LassoCV

# 1. Synthesize 50 features where only 5 carry true signal, 45 are pure noise
X, y = make_classification(n_samples=500, n_features=50, n_informative=5, random_state=42)

# --- Method 1: Filter Method (Mutual Information) ---
selector_mi = SelectKBest(score_func=mutual_info_classif, k=5)
X_mi = selector_mi.fit_transform(X, y)
selected_mi_idx = np.where(selector_mi.get_support())[0]
print(f"Filter Method (Mutual Info) Selected Features: {selected_mi_idx}")

# --- Method 2: Wrapper Method (Recursive Feature Elimination - RFE) ---
estimator = LogisticRegression(max_iter=1000)
rfe = RFE(estimator=estimator, n_features_to_select=5, step=5)
X_rfe = rfe.fit_transform(X, y)
selected_rfe_idx = np.where(rfe.support_)[0]
print(f"Wrapper Method (RFE) Selected Features:        {selected_rfe_idx}")

# --- Method 3: Embedded Method (L1 Lasso Regularization) ---
lasso = LassoCV(cv=5, random_state=42).fit(X, y)
selected_lasso_idx = np.where(np.abs(lasso.coef_) > 1e-4)[0]
print(f"Embedded Method (L1 Lasso) Selected Features:   {selected_lasso_idx}")
```

---

## 5. Demystifying the Mathematics: Explaining Every Formula

Let's dissect the mathematical equations governing feature engineering and selection.

### Equation 1: Mutual Information Formula
How do we mathematically measure how much information feature $X$ carries about target $Y$?

$$
I(X; Y) = \sum_{x \in X} \sum_{y \in Y} P(x, y) \log \left( \frac{P(x, y)}{P(x) P(y)} \right)
$$

- $P(x, y)$: The **Joint Probability** of observing $X=x$ and $Y=y$ together.
- $P(x) P(y)$: The expected probability if $X$ and $Y$ were completely independent.
- If $X$ and $Y$ are independent, $P(x, y) = P(x)P(y) \to \frac{P(x, y)}{P(x)P(y)} = 1 \to \log(1) = 0$. Mutual information is zero!

---

### Equation 2: Cyclical Sin/Cos Transformation
How do we map hour $t \in [0, 23]$ onto a continuous 2D circle?

$$
x_{\sin} = \sin\left(\frac{2\pi \cdot t}{T}\right), \quad x_{\cos} = \cos\left(\frac{2\pi \cdot t}{T}\right)
$$

- $t$: Current hour value ($0 \dots 23$).
- $T$: Total cycle length ($T = 24$ hours).
- $\frac{2\pi \cdot t}{T}$: Converts the hour index into an angle in radians ($0 \to 2\pi$). Taking $\sin$ and $\cos$ yields 2D coordinates on a circle where Hour 23 and Hour 0 sit right next to each other!

---

## 6. Real-World Production Gotchas & Failure Modes

### Multicollinearity & Variance Inflation Factor (VIF)
Including highly correlated features (e.g., `height_inches` and `height_cm`) inflates the variance of linear model coefficients, making weight values unstable.
- *Fix*: Calculate Variance Inflation Factor (VIF) and drop features with $\text{VIF} > 10$:
  $$
  \text{VIF}_j = \frac{1}{1 - R_j^2}
  $$

---

## 7. Feynman Exercises & Deep Thinking Challenges

1. **Thought Experiment**: Why does L1 Lasso regularization drive non-essential feature weights to *exact zero*, whereas L2 Ridge regularization only shrinks weights close to zero?
   - *Answer/Explanation*: L1 regularization adds an absolute value penalty $|w|$, which has constant slope gradients down to zero. Geometrically, the L1 constraint region is a diamond with sharp corners on axes, causing optimal solution contours to intersect exactly at zero.

2. **Code Challenge**: Evaluate feature importance using `RFE` with `step=1` versus `step=10` on a 200-feature dataset to observe speed vs. accuracy trade-offs.
