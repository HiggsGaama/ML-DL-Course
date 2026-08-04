# Module 03: Linear & Logistic Regression

## 1. The Physical Intuition: Sliding Planes and Probability Gates

Imagine you are standing on a hilly landscape in pitch-black darkness. You cannot see where the bottom of the valley is. All you can feel under your boots is the local slope of the ground. How do you find the absolute lowest point in the valley? 

Simple! You take a step in whatever direction feels steepest downhill. Then you check the slope again under your boots and take another step downhill. You repeat this process over and over until the ground under your feet feels completely flat. Congratulations: you've just executed **Gradient Descent**!

```
Linear Regression:   Draw the smoothest straight line/plane through data points.
Logistic Regression: Take that straight line and bend it through a S-curve gate to output probabilities between 0 and 1.
```

Linear Regression and Logistic Regression are the twin bedrock foundational algorithms of machine learning:
- **Linear Regression**: Predicts a continuous scalar number ($y \in [-\infty, +\infty]$). "If I increase advertising spend by $\$1,000$, how many dollars of product will I sell?"
- **Logistic Regression**: Predicts a probability confidence score ($p \in [0.0, 1.0]$). "Given a credit card transaction's location and amount, what is the probability this is fraudulent?"

---

## 2. Core Mechanics & Granular Step-by-Step Breakdown

### 1. Linear Regression: Fitting the Hyperplane

Imagine scattering 100 ping-pong balls floating in 3D room space. You want to insert a flat sheet of stiff cardboard through the middle of the cluster so that the total distance from all balls to the cardboard sheet is as small as possible.

```
       y ^                                          y ^
         │        •                                   │        • /
         │     •                                      │     •  /
         │  •                                         │  •   /
         │ •                                          │ •  /
         └───────────────────► x                      └───────────────────► x
           (Unfitted Data Points)                       (Fitted Best Line: y = wx + b)
```

The mathematical equation of that cardboard plane is:

$$\hat{y} = w_1 x_1 + w_2 x_2 + b$$

Where:
- $x_1, x_2$ are your input features (e.g., $x_1 = \text{Square Footage}, x_2 = \text{Bedrooms}$).
- $w_1, w_2$ are the **Weights** determining the tilt/slope of the plane along each feature axis.
- $b$ is the **Bias** (the height where the plane crosses the vertical $y$-axis).

To fit this plane, we measure the total error using **Mean Squared Error (MSE)**. We then use **Gradient Descent** to iteratively tilt $w$ and slide $b$ until the error reaches its lowest possible valley!

---

### 2. Logistic Regression: The Sigmoid Probability Gate

Now, what if your target $y$ isn't a continuous height, but a binary decision: `0 (Healthy)` vs. `1 (Sick)`?

If you try to fit a straight line to binary $0/1$ data, two terrible things happen:
1. A straight line keeps going forever! It will predict values like $\hat{y} = 1.8$ or $\hat{y} = -0.4$, which makes zero sense as probabilities.
2. An extreme outlier data point far to the right pulls the entire line down, ruining predictions for normal points!

```
Straight Line on Binary Data (FAIL):         Sigmoid S-Curve (SUCCESS):
y ^                                          y ^
1 ┼        •  •  •  • (Class 1)              1 ┼ - - - - - - - - . - - - - - (Class 1)
  │       /                                    │               . '
  │      /                                 0.5 ┼ - - - - - . '  <-- Decision Boundary
  │     /                                      │         . '
0 ┼  • / •  • (Class 0)                      0 ┼ . - - ' - - - - - - - - - - (Class 0)
  └────────────────────────► x                 └────────────────────────► z = w^T x + b
```

To fix this, we take our linear score $z = w^T x + b$ and pass it through a mathematical transformation function called the **Sigmoid Function** $\sigma(z)$:

$$\hat{p} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Look at what this magical function does:
- If $z$ is a huge positive number (e.g., $z = +10$), $e^{-10} \approx 0$, making $\hat{p} = \frac{1}{1 + 0} = 1.0$.
- If $z$ is a huge negative number (e.g., $z = -10$), $e^{+10}$ is massive, making $\hat{p} = \frac{1}{1 + \text{huge}} = 0.0$.
- If $z = 0$, $e^{0} = 1$, making $\hat{p} = \frac{1}{1 + 1} = 0.5$.

The Sigmoid function acts like a **smooth mathematical gate**, bending an infinite straight line into a beautiful S-curve locked strictly between $0.0$ and $1.0$!

---

### 3. The Decision Boundary
By default, if $\hat{p} \ge 0.5$, we predict Class `1`; if $\hat{p} < 0.5$, we predict Class `0`.
Since $\sigma(z) = 0.5$ when $z = 0$, the line or plane defined by $w^T x + b = 0$ serves as the exact **Decision Boundary** dividing the two classes in feature space.

---

## 3. Architecture & Visual Diagrams

### Gradient Descent Optimization Surface

```
  Weight w_2
     ^
     │      (Convex Bowl Loss Contours J(w))
     │           . ─── .
     │        . '       ' .
     │       /   (Min)    \
     │      |     x        |   <-- Global Loss Minimum
     │       \            /
     │        ' .      . '
     │           ' ── '  ◄────── Starting Point (Initial Random Weights)
     │            \
     │             └─── Step 1 ──► Step 2 ──► Step 3 (Steepest Downhill Gradient Steps)
     └─────────────────────────────────────────────────────────────► Weight w_1
```

---

## 4. Practical Implementation: Training Linear and Logistic Models

Let's write a clean Python script using `scikit-learn` to demonstrate both Linear Regression and Logistic Regression, inspecting learned weights, probabilities, and decision boundaries:

```python
import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, log_loss

# ==========================================
# PART 1: LINEAR REGRESSION EXPERIMENT
# ==========================================
print("=== PART 1: LINEAR REGRESSION ===")
# Synthesize physical data: y = 4.5 * x1 - 2.0 * x2 + 10.0 + noise
X_reg, y_reg = make_regression(n_samples=200, n_features=2, noise=5.0, random_state=42)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

lin_model = LinearRegression()
lin_model.fit(X_train_r, y_train_r)

y_pred_r = lin_model.predict(X_test_r)

print(f"True Physics Weights:  w1 ≈ 4.5, w2 ≈ -2.0, b ≈ 10.0")
print(f"Learned Weights (w1, w2): {lin_model.coef_.round(4)}")
print(f"Learned Bias    (b):      {lin_model.intercept_:.4f}")
print(f"Test Set MSE Loss:        {mean_squared_error(y_test_r, y_pred_r):.4f}")
print(f"R-squared Variance Score: {r2_score(y_test_r, y_pred_r):.4f}")

# ==========================================
# PART 2: LOGISTIC REGRESSION EXPERIMENT
# ==========================================
print("\n=== PART 2: LOGISTIC REGRESSION ===")
# Synthesize binary classification data
X_cls, y_cls = make_classification(n_samples=300, n_features=2, n_redundant=0, random_state=42)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

log_model = LogisticRegression(C=1.0) # C = 1/lambda inverse regularization
log_model.fit(X_train_c, y_train_c)

y_pred_c = log_model.predict(X_test_c)
y_prob_c = log_model.predict_proba(X_test_c)[:, 1] # Probability of Class 1

print(f"Learned Classifier Weights: {log_model.coef_[0].round(4)}")
print(f"Learned Classifier Bias:    {log_model.intercept_[0]:.4f}")
print(f"Test Set Accuracy:          {accuracy_score(y_test_c, y_pred_c):.4f}")
print(f"Test Set Binary Log-Loss:   {log_loss(y_test_c, y_prob_c):.4f}")

# Debug individual probability prediction
sample_idx = 0
print(f"\nDebugging Sample #{sample_idx}:")
print(f"  -> Calculated Logit Score (z): {log_model.decision_function(X_test_c)[sample_idx]:.4f}")
print(f"  -> Sigmoid Probability (p):   {y_prob_c[sample_idx]:.4f}")
print(f"  -> Final Assigned Class:      {y_pred_c[sample_idx]} (Actual: {y_test_c[sample_idx]})")
```

---

## 5. Demystifying the Mathematics: Explaining Every Formula

Let's dissect the core mathematical equations governing optimization, loss functions, and probability transformations.

### Equation 1: The Sigmoid Function
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

- $z$: The unconstrained linear combination score ($z = w^T x + b = w_1 x_1 + \dots + w_d x_d + b$).
- $e^{-z}$: Euler's constant ($e \approx 2.71828$) raised to the power of negative $z$. 
  - As $z$ grows positive, $e^{-z} \to 0$, forcing $\sigma(z) \to \frac{1}{1 + 0} = 1$.
  - As $z$ drops negative, $e^{-z} \to \infty$, forcing $\sigma(z) \to \frac{1}{1 + \infty} = 0$.

---

### Equation 2: Binary Cross-Entropy Loss (Log-Loss)
Why can't we use Mean Squared Error for Logistic Regression? Because passing the non-linear Sigmoid into MSE creates a bumpy, non-convex loss landscape with hundreds of local fake bottoms. Gradient descent would get stuck in a bad local minimum!

Instead, we use **Binary Cross-Entropy Loss (Log-Loss)**, which creates a smooth, perfectly convex bowl:

$$
J(w, b) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y^{(i)} \log\left(\hat{p}^{(i)}\right) + \left(1 - y^{(i)}\right) \log\left(1 - \hat{p}^{(i)}\right) \right]
$$

Let's dissect this terrifying-looking equation to see how simple it actually is!
Notice that target $y^{(i)}$ is always either $1$ or $0$. Look at what happens to the formula in both cases:

1. **Case A: When True Target $y^{(i)} = 1$**
   - The second term $(1 - y^{(i)})\log(1 - \hat{p}^{(i)})$ becomes $(1 - 1) = 0$, vanishing completely!
   - The loss simplifies to: $\text{Loss} = -\log(\hat{p}^{(i)})$.
   - If the model correctly predicts $\hat{p} = 1.0$, $\log(1.0) = 0 \to \text{Loss} = 0$.
   - If the model confidently lies and predicts $\hat{p} = 0.001$, $\log(0.001) = -6.9 \to \text{Loss} = +6.9$ (huge penalty!).

2. **Case B: When True Target $y^{(i)} = 0$**
   - The first term $y^{(i)}\log(\hat{p}^{(i)})$ becomes $0 \cdot \log(\hat{p}) = 0$, vanishing completely!
   - The loss simplifies to: $\text{Loss} = -\log(1 - \hat{p}^{(i)})$.
   - If the model correctly predicts $\hat{p} = 0.0$, $\log(1.0) = 0 \to \text{Loss} = 0$.
   - If the model lies and predicts $\hat{p} = 0.999$, $\log(0.001) = -6.9 \to \text{Loss} = +6.9$!

The minus sign at the front $-\frac{1}{n}$ turns negative logarithms into positive loss penalties. It acts like a strict referee punishing confident lies!

---

### Equation 3: Gradient Descent Update Step
How does the computer update its weights step-by-step?

$$
\frac{\partial J}{\partial w} = \frac{1}{n} X^T (\hat{y} - y)
$$

$$
w \leftarrow w - \alpha \frac{\partial J}{\partial w}
$$

- $\frac{\partial J}{\partial w}$: The **Gradient** (partial derivative). It measures the local slope of the loss landscape with respect to each weight $w_j$.
- $(\hat{y} - y)$: The error residual vector.
- $\alpha$ (alpha): The **Learning Rate**. This controls step size:
  - If $\alpha$ is too small (e.g., $0.000001$), training takes 3 days.
  - If $\alpha$ is too large (e.g., $10.0$), the model takes giant leaps over the valley, exploding upward into infinity!

---

## 6. Real-World Production Gotchas & Failure Modes

### 1. Numerical Instability in Log-Loss (`log(0)` Panic)
If your model predicts an exact probability of $\hat{p} = 0.0$ for a sample whose true target is $y = 1$, evaluating $\log(0)$ produces `NaN` or `-inf`. This crashes your PyTorch training loop instantly!
- *Fix*: Always clamp predicted probabilities inside a tiny safe margin:
  ```python
  p_clamped = np.clip(p_predicted, 1e-15, 1.0 - 1e-15)
  ```

### 2. Multi-Collinearity Explosion
If you pass two features that are virtually identical (e.g., `Distance_Miles` and `Distance_KM`), linear models get confused. Weight $w_1$ could jump to $+1,000,000$ while weight $w_2$ drops to $-1,600,000$, canceling each other out while making predictions unstable!
- *Fix*: Use L2 Ridge Regularization ($J(w) + \lambda \|w\|_2^2$), which penalizes massive weights.

---

## 7. Feynman Exercises & Deep Thinking Challenges

1. **Thought Experiment**: What happens to the Sigmoid curve $\sigma(z) = \frac{1}{1 + e^{-(w x + b)}}$ as weight $w$ becomes extremely huge ($w = +1000$)? What happens as $w \to 0$?
   - *Answer/Explanation*: As $w \to \infty$, the Sigmoid curve becomes a sharp, vertical **step function** (hard threshold at $x = -b/w$). As $w \to 0$, the curve flattens into a horizontal line at $p = 0.5$, completely ignoring input $x$!

2. **Code Challenge**: Take the Logistic Regression code in Section 4. Intentionally set learning rate $\alpha = 100.0$ in a manual Gradient Descent loop. Watch how the loss explodes instead of descending!
