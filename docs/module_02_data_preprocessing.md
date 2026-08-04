# Module 02: Data Preprocessing & Sanitization Pipelines

## 1. The Physical Intuition: Distilling Ore into Pure Metal

Imagine you are a chemical engineer building a high-precision battery manufacturing plant. Someone dumps a truckload of unrefined raw earth in front of your factory: chunks of rock, damp mud, copper flecks, dead leaves, and plastic trash. 

Can you feed raw dirt directly into an automated battery assembly machine? Of course not! The machine expects pure, ultra-refined chemical elements—lithium, nickel, cobalt—molded into standardized shapes with micro-millimeter precision.

In Machine Learning, **raw real-world data is unrefined dirt**. It arrives from SQL databases, web scraping scripts, or user inputs filled with:
- Missing entries (`NaN`, `null`, empty strings).
- Heterogeneous text labels (`"New York"`, `"LA"`, `"Chicago"`).
- Vastly mismatched numeric scales (`Age: 18 - 75` vs. `Annual Income: $20,000 - $1,000,000`).

Mathematical models—whether linear regressions, support vector machines, or 100-layer neural networks—are purely numeric engines. They perform linear algebra operations on matrices of floating-point numbers ($\mathbb{R}^{n \times d}$). 

**Data Preprocessing is your chemical refinery**: the systematic ETL pipeline that cleans, encodes, and rescales messy raw data into a pristine, normalized numerical tensor matrix.

```
[ Raw Dirty Data ] (SQL/CSV with NaNs, Text Strings, Unscaled Magnitudes)
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ Production Preprocessing Refinery                      │
│  1. Missing Value Imputation (NaN -> Median / Mode)    │
│  2. Categorical Encoding (Strings -> Numeric Vectors)   │
│  3. Feature Scaling (Standardizing Magnitudes)         │
└────────────────────────────────────────────────────────┘
          │
          ▼
[ Clean Floating-Point Matrix X ∈ ℝ^(n x d) ] ──► Fed into Machine Learning Engine
```

---

## 2. Unpacking the Three Pillars of Data Refinement

### 1. Handling Missing Data (Imputation): Filling the Holes

What happens when a sensor drops connection for 5 minutes, or a user skips an optional questionnaire field? Your DataFrame gets filled with `NaN` (Not a Number) values.

If you pass a `NaN` into a mathematical dot product $w_1 x_1 + w_2 x_2$, the result becomes `NaN`. The error propagates through every matrix operation, turning your entire model into garbage!

```
                                HANDLING MISSING DATA
                                          │
       ┌──────────────────────────────────┼──────────────────────────────────┐
       ▼                                  ▼                                  ▼
[ Drop Records ]                 [ Simple Imputation ]             [ Advanced Model Imputation ]
Delete rows with NaNs            Replace with Mean/Median/Mode    Predict NaNs using KNN/MICE
(Risks losing data)              (Fast, robust baseline)          (Accurate, higher compute)
```

1. **Strategy A: Record Deletion (Dropping Rows)**
   - You simply delete any row that contains a `NaN`.
   - *The Trap*: If $5\%$ of values are missing across 20 columns randomly, you might end up throwing away $60\%$ of your entire dataset! Use dropping ONLY if missingness is $<1\%$.

2. **Strategy B: Simple Statistical Imputation**
   - **Mean Imputation**: Replaces `NaN` with the average of the column. *Warning*: Highly vulnerable to extreme outliers! (If one user earns $\$100\text{M}$, the mean income skyrockets, filling missing values with unrealistic numbers).
   - **Median Imputation**: Replaces `NaN` with the 50th percentile (middle value). *Extremely robust* against outliers. This is the gold-standard default for numerical data.
   - **Mode Imputation**: Replaces `NaN` with the most frequently occurring value. Perfect for categorical strings like `Color: "Red"`.

3. **Strategy C: Advanced Predictive Imputation (KNN Imputer)**
   - Finds the $K$ complete rows in your dataset that are most similar to the incomplete row, and takes the average of *their* values to fill the hole.

---

### 2. Categorical Encoding: Translating Words to Numbers

Computers cannot multiply the string `"New York"` by a weight vector $w$. We must translate human text labels into numbers. But beware! How you translate words fundamentally alters how the computer understands spatial relationships.

```
Original Feature: City = ["NYC", "LA", "Chicago"]

WRONG APPROACH (Ordinal Encoding Nominal Data):
  "NYC" -> 0, "LA" -> 1, "Chicago" -> 2
  Problem: Model assumes Chicago (2) is TWICE as large as LA (1)!

RIGHT APPROACH (One-Hot Encoding Nominal Data):
  "NYC"     -> [1, 0, 0]
  "LA"      -> [0, 1, 0]
  "Chicago" -> [0, 0, 1]
```

1. **One-Hot Encoding (OHE)**
   - Creates a new binary $(0 \text{ or } 1)$ indicator column for every unique category.
   - *When to use*: Nominal categories with no intrinsic mathematical order (e.g., `Cities`, `Colors`, `Operating Systems`).

2. **Ordinal Encoding**
   - Assigns sequential integers ($0, 1, 2, 3 \dots$) to categories.
   - *CRITICAL RULE*: Use ONLY when categories possess a genuine, inherent mathematical ranking! For example: `Education Level: ["High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3]`. Here, $3 > 2 > 1 > 0$ reflects real ordinal structure.

3. **Target Encoding**
   - Replaces each category string with the average target output value $\bar{y}$ for that specific category.
   - *When to use*: High-cardinality features where One-Hot Encoding would create 50,000 sparse columns (e.g., `ZIP Codes`, `User IDs`).

---

### 3. Feature Scaling: Balancing the Physical Magnitudes

Imagine you are building a model to predict health risks using two features: `Age` (ranging from $18$ to $80$) and `Annual Income` (ranging from $\$20,000$ to $\$500,000$).

Now, think about what happens when a distance-based model (like KNN or Support Vector Machines) computes the geometric Euclidean distance between Person A ($\text{Age}=25, \text{Income}=\$50,000$) and Person B ($\text{Age}=55, \text{Income}=\$100,000$):

$$
d = \sqrt{(55 - 25)^2 + (100000 - 50000)^2} = \sqrt{30^2 + 50000^2} = \sqrt{900 + 2,500,000,000} \approx 50,000.009
$$

Look at those numbers! The difference in age ($30^2 = 900$) is completely swallowed by the difference in income ($50000^2 = 2.5 \times 10^9$). The computer completely ignores `Age` because its raw numbers are smaller! 

To fix this visual distortion, we must **Scale** our features so they occupy comparable numerical ranges.

```
Raw Unscaled Space (Income dominates Age):
Income ^
 $500k │                     • Person B
       │
       │
  $50k │    • Person A
       └───────────────────────────────────► Age
           18                            80

Scaled Space (Equal geometric weighting):
Standardized ^
    +2.0     │                     • Person B
      0      ├───────── • Center
    -2.0     │    • Person A
             └─────────────────────────────► Standardized
                -2.0       0      +2.0
```

1. **Standard Scaling (Z-Score Normalization)**
   - Centers data so the mean becomes $\mu = 0$ and standard deviation becomes $\sigma = 1$.
   - *Best for*: Neural networks, logistic regression, and algorithms assuming Gaussian distributions.

2. **Min-Max Scaling**
   - Squeezes every single value strictly into a fixed box between $0.0$ and $1.0$.
   - *Danger*: Highly vulnerable to outliers! If one person earns $\$100,000,000$, Min-Max scaling compresses all normal people into a tiny cluster between $0.0001$ and $0.001$.

3. **Robust Scaling**
   - Uses the **Median** and **Interquartile Range (IQR)** instead of mean and standard deviation.
   - *Best for*: Messy real-world datasets containing extreme outliers.

---

## 3. Visual Architecture: The ColumnTransformer Pipeline

In production code, you assemble these steps into an integrated processing pipeline:

```
                                  RAW INPUT DATASET (X_train)
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      │                                               │
             Numerical Columns                               Categorical Columns
        ['age', 'income', 'credit_score']                     ['country', 'device']
                      │                                               │
                      ▼                                               ▼
         ┌─────────────────────────┐                     ┌─────────────────────────┐
         │  Numerical Pipeline     │                     │  Categorical Pipeline   │
         │  1. Median Imputer      │                     │  1. Mode Imputer        │
         │  2. StandardScaler      │                     │  2. OneHotEncoder       │
         └────────────┬────────────┘                     └────────────┬────────────┘
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              │
                                              ▼
                              [ Feature Union / Concatenation ]
                                              │
                                              ▼
                               PROCESSED NUMERICAL TENSOR MATRIX
                                     X_processed ∈ ℝ^(n x d)
```

---

## 4. Practical Implementation: Building a Production ETL Pipeline

Let's write a complete, runnable Python script using `scikit-learn` that cleans missing values, encodes categories, scales numbers, and guards against data leakage:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Create a raw, messy dataset with missing values (NaNs) and mixed data types
raw_data = pd.DataFrame({
    'age': [25.0, np.nan, 47.0, 52.0, 31.0, 22.0, 38.0, 61.0, np.nan, 29.0],
    'income': [50000, 64000, 120000, np.nan, 80000, 45000, 95000, 150000, 52000, 68000],
    'education': ['Bachelors', 'Masters', 'PhD', 'High School', 'Bachelors', 'High School', 'Masters', 'PhD', 'Bachelors', 'Masters'],
    'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA', 'Chicago', 'NYC', 'LA', 'Chicago', 'NYC'],
    'bought_product': [0, 1, 0, 1, 0, 0, 1, 1, 0, 0]
})

X = raw_data.drop(columns=['bought_product'])
y = raw_data['bought_product']

# ABSOLUTE RULE: Split into Train and Test BEFORE applying any preprocessing steps!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Define independent sub-pipelines for numerical vs. categorical features
numerical_cols = ['age', 'income']
categorical_cols = ['education', 'city']

# Numerical Pipeline: Fill NaNs with Median -> Scale to Mean=0, Std=1
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical Pipeline: Fill NaNs with Mode -> One-Hot Encode strings
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

# 3. Combine sub-pipelines using ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num_transform', num_pipeline, numerical_cols),
    ('cat_transform', cat_pipeline, categorical_cols)
])

# 4. Wrap Preprocessor + Classifier into one unified execution model
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# 5. Fit ONCE on X_train (Learns median, mean, std, and one-hot categories from train set ONLY)
full_pipeline.fit(X_train, y_train)

# 6. Predict on unseen X_test (Applies learned parameters without re-fitting)
predictions = full_pipeline.predict(X_test)

print("=== PREPROCESSING & TRAINING COMPLETE ===")
print(f"Test Set Accuracy: {accuracy_score(y_test, predictions):.4f}")

# Inspect transformed matrix shape
X_train_transformed = preprocessor.transform(X_train)
print(f"Transformed Training Feature Matrix Shape: {X_train_transformed.shape}")
print("Notice how categorical text strings became clean one-hot floating point numbers!")
```

---

## 5. Demystifying the Mathematics: Explaining Every Formula

Let's break down the exact mathematical equations governing feature scaling.

### Equation 1: Standard Scaler (Z-Score Normalization)
How do you transform any dataset so its mean becomes $0$ and its standard deviation becomes $1$?

$$
z = \frac{x - \mu}{\sigma}
$$

Let's dissect every piece of this formula:
- $x$: The raw feature value of an individual sample (e.g., $x = \$80,000$ income).
- $\mu$ (mu): The average (mean) of all values in that column: $\mu = \frac{1}{n}\sum_{i=1}^n x_i$.
- $x - \mu$: The **Deviation**. By subtracting the average $\mu$ from every value $x$, the new center of the data shifts to $0.0$. Values above average become positive; values below average become negative.
- $\sigma$ (sigma): The **Standard Deviation**, which measures the average spread/dispersion of the data:
  $$
  \sigma = \sqrt{\frac{1}{n}\sum_{i=1}^n (x_i - \mu)^2}
  $$
- Dividing by $\sigma$: Dividing the deviation by $\sigma$ rescales the spread. A final Z-score of $z = +2.0$ means *"this specific data point sits exactly 2 standard deviations above the population average."*

---

### Equation 2: Min-Max Scaling
How do you squeeze every number inside a strict boundary box between $0.0$ and $1.0$?

$$
x_{\text{scaled}} = \frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}}
$$

- $x_{\text{min}}$: The smallest number in the entire column.
- $x_{\text{max}}$: The largest number in the entire column.
- $x - x_{\text{min}}$: Distance from the minimum boundary. If $x = x_{\text{min}}$, the numerator becomes $0$, making $x_{\text{scaled}} = 0.0$.
- $x_{\text{max}} - x_{\text{min}}$: Total range (span) of the column data. If $x = x_{\text{max}}$, the numerator equals the denominator, making $x_{\text{scaled}} = 1.0$.

---

### Equation 3: Robust Scaler (Outlier-Proof Scaling)
When extreme outliers corrupt the mean $\mu$ and range, we use quartiles instead:

$$
x_{\text{robust}} = \frac{x - Q_2(x)}{Q_3(x) - Q_1(x)}
$$

- $Q_2(x)$: The **Median** (50th percentile). Half the data sits above it, half below.
- $Q_1(x)$: The 25th percentile (bottom quartile boundary).
- $Q_3(x)$: The 75th percentile (top quartile boundary).
- $Q_3(x) - Q_1(x)$: The **Interquartile Range (IQR)**, representing the spread of the middle 50% of your data. Outliers hanging out in extreme tails have zero influence on $Q_1, Q_2,$ or $Q_3$!

---

## 6. Real-World Production Gotchas & Failure Modes

### 1. The Dummy Variable Trap (Perfect Multicollinearity)
If you have a feature `Gender: ["Male", "Female"]` and you One-Hot Encode it into two columns:
- `Is_Male: [1, 0, 1]`
- `Is_Female: [0, 1, 0]`

Notice that $\text{Is\_Male} + \text{Is\_Female} = 1$ always! One column is a perfect linear prediction of the other ($\text{Is\_Female} = 1 - \text{Is\_Male}$). This breaks linear regression algorithms because it makes matrix inversion impossible (singular matrix error).
- *Fix*: Always drop one binary column using `drop='first'`. For $k$ categories, create $k-1$ columns.

### 2. The Unseen Category Production Crash
Your model trains on `City: ["NYC", "LA"]`. Two months later in production, a user from `"Seattle"` submits a form. Standard one-hot encoders throw an unhandled `ValueError`, crashing your API server!
- *Fix*: Always instantiate `OneHotEncoder(handle_unknown='ignore')`. When an unseen string appears, it encodes it cleanly as all zeros (`[0, 0]`).

---

## 7. Feynman Exercises & Deep Thinking Challenges

1. **Thought Experiment**: Suppose you have a dataset where 90% of a feature's values are $0$, and 10% are $1,000,000$. If you apply `StandardScaler`, what happens to the mean and standard deviation? What will the scaled value of $0$ become?
   - *Answer/Explanation*: The mean is $\mu = 100,000$. The raw value $0$ becomes negative: $z = \frac{0 - 100000}{\sigma} \approx -0.33$. Notice that standard scaling turns sparse zero-filled matrices into dense non-zero matrices! For sparse matrices, use `MaxAbsScaler` instead.

2. **Code Challenge**: Take the script from Section 4. Intentionally add an extreme outlier (`income = 1,000,000,000`). Replace `StandardScaler` with `MinMaxScaler` and print the transformed outputs. Then swap in `RobustScaler`. Observe how `RobustScaler` keeps normal incomes well-spaced while `MinMaxScaler` crushes them all to zero!
