# Module 07: Hyperparameter Tuning & Optuna Optimization

## 1. The Physical Intuition: Tuning a High-Performance Engine

Imagine you are a Formula 1 race car engineer. You are handed a high-performance race car engine, but it comes with 15 adjustable mechanical knobs: fuel-to-air mixture ratio, turbocharger pressure, suspension stiffness, tire inflation pressure, and gear ratio splits.

The engine doesn't tune its own mechanical knobs while driving down the track. The engine's internal pistons and crankshaft (the **Parameters**, like weights $w$ and biases $b$) move automatically during operation. But YOU must set the external control knobs (the **Hyperparameters**) before the car leaves the pit lane!

If you set the turbocharger pressure too high, the engine explodes on lap 3 (Overfitting / Training Crash). If you set it too low, the car chokes on straightaways (Underfitting).

```
Parameters (Learned Automatically by Model):   Weights w, Biases b, Tree Split Thresholds
Hyperparameters (Configured by Engineer):      Learning Rate, Tree Depth, Regularization C, Estimators
```

**Hyperparameter Tuning** is the science of searching the multidimensional configuration landscape to discover control knob settings that unlock peak performance without crashing the system.

---

## 2. Core Mechanics & Granular Step-by-Step Breakdown

### 1. The Evolution of Tuning Strategies

```
GRID SEARCH:            RANDOM SEARCH:          BAYESIAN OPTIMIZATION (OPTUNA):
Exhaustive grid map     Random uniform guesses  Smart probabilistic search
(Combinatorial explosion) (Efficient exploration) (Learns from past trials + early pruning)
```

1. **Grid Search (`GridSearchCV`)**:
   - You define a fixed grid: `max_depth = [3, 5, 10]` $\times$ `learning_rate = [0.01, 0.1, 0.2]`.
   - The algorithm exhaustively tries all $3 \times 3 = 9$ combinations.
   - *The Catch*: Exponential combinatorial explosion ($O(K^d)$). Trying 5 parameters with 10 values each requires $10^5 = 100,000$ cross-validated model runs!

2. **Random Search (`RandomSearchCV`)**:
   - Randomly samples $N$ trials from specified probability distributions.
   - *Key Insight*: Far more efficient than Grid Search when 1 or 2 hyperparameters dominate model performance while others exert minor influence.

3. **Bayesian Optimization & Optuna (Tree-structured Parzen Estimator - TPE)**:
   - Instead of guessing blindly, Optuna builds a **probabilistic surrogate model** of historical trial outcomes. It asks: *"Based on the 20 trials I ran previously, which exact hyperparameter region has the highest mathematical probability of improving my validation score?"*
   - **Early Pruning (Median Pruner)**: If a trial performs terribly on epoch 3, Optuna kills it immediately without wasting compute time on epochs 4–100!

---

## 3. Visual Architecture: Optuna Smart Optimization Loop

```
┌────────────────────────────────────────────────────────┐
│ 1. Optuna Sampler proposes candidate parameters θ       │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ 2. Start Training Model on Cross-Validation Fold       │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
                  Is Intermediate Score
                 Worse Than Historical Median?
                  /              \
           YES   /                \  NO
                ▼                  ▼
     ┌────────────────────┐   ┌──────────────────────────┐
     │ PRUNE TRIAL EARLY! │   │ Complete Full Evaluation │
     │ (Save Compute Time)│   │ Return Validation Metric │
     └────────────────────┘   └────────────┬─────────────┘
                                           │
                                           ▼
                               Update Bayesian Surrogate
```

---

## 4. Practical Implementation: Optuna Hyperparameter Optimization

Let's write a clean Python script using **Optuna** to optimize an XGBoost Classifier with automated trial pruning:

```python
import optuna
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier

# Quiet down Optuna info logs for clean output
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 1. Synthesize Benchmark Classification Dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)

# 2. Define Objective Function for Optuna
def objective(trial):
    # Define Hyperparameter Search Space with appropriate sampling scales
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True), # Logarithmic sampling!
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'random_state': 42
    }
    
    model = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores.mean()

# 3. Create Study Engine & Execute Optimization
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner()
)
study.optimize(objective, n_trials=30, timeout=60)

# 4. Display Results
print("=== OPTUNA BAYESIAN TUNING COMPLETE ===")
print(f"Best Trial Accuracy Score: {study.best_value:.4f}")
print("Best Discovered Hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key:<20}: {value}")
```

---

## 5. Demystifying the Mathematics: Explaining Every Formula

Let's dissect how Optuna's Tree-structured Parzen Estimator (TPE) decides which point to test next.

### Equation 1: Expected Improvement (EI)
$$
\text{EI}(x) = \int_{-\infty}^{y^*} (y^* - y) \, p(y \mid x) \, dy
$$

- $x$: A candidate hyperparameter configuration.
- $y^*$: The current best validation score threshold achieved across past runs.
- $(y^* - y)$: The improvement magnitude over our previous record score.
- $p(y \mid x)$: The probability distribution predicting what score configuration $x$ will yield.

---

### Equation 2: TPE Probability Density Ratio
Instead of modeling $p(y \mid x)$ directly, TPE uses Bayes' Rule to split past trials into two probability density groups:

$$
p(x \mid y) = \begin{cases} 
l(x) & \text{if } y < y^* \quad (\text{Good Trials - Top 15\%}) \\
g(x) & \text{if } y \ge y^* \quad (\text{Bad Trials - Remaining 85\%})
\end{cases}
$$

$$\text{EI}(x) \propto \frac{l(x)}{g(x)}$$

- $l(x)$: Density function of hyperparameter choices that led to **good** scores.
- $g(x)$: Density function of hyperparameter choices that led to **bad** scores.
- **The Optimization Strategy**: Optuna chooses candidate points $x$ that maximize the ratio $\frac{l(x)}{g(x)}$—picking parameters that are highly likely under good trials and highly unlikely under bad trials!

---

## 6. Real-World Production Gotchas & Failure Modes

### Sampling Scale Bug (Linear vs. Logarithmic Scale)
If you sample a parameter spanning orders of magnitude (like learning rate `0.0001` to `0.1`) on a linear scale, 90% of samples fall between `0.01` and `0.1`, ignoring tiny rates!
- *Fix*: Always set `log=True` when sampling rates or regularization parameters across multiple orders of magnitude.

---

## 7. Feynman Exercises & Deep Thinking Challenges

1. **Thought Experiment**: Why does Nested Cross-Validation prevent optimistic hyperparameter tuning bias?
   - *Answer/Explanation*: Standard CV selects parameters optimized for that specific validation split, creating slight overfitting to the validation data. Nested CV uses an Inner Loop for parameter selection and an Outer Loop for unbiased evaluation on un-tuned holdout folds.

2. **Code Challenge**: Add `optuna.pruners.MedianPruner()` to your script and compare execution time against unpruned search.
