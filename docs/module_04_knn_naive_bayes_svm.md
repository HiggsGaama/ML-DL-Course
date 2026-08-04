# Module 04: KNN, Naive Bayes & Support Vector Machines (SVM)

## 1. The Physical Intuition: Three Different Ways to See the World

Imagine walking into a room full of strangers at a party, and you want to guess what kind of music a person named Alex likes. How would you solve this problem?

You have three completely different ways of thinking available to you:

1. **The Lazy Neighbor Strategy (K-Nearest Neighbors - KNN)**: Look around the room, find the 5 people who dress and talk most similarly to Alex, and ask what music *they* like. If 4 of them say "Jazz", you guess that Alex likes Jazz. You haven't built a theory of music preference; you just looked at physical neighbors.

2. **The Frequency Counting Strategy (Naive Bayes)**: You recall past statistics. "Given that Alex is wearing a leather jacket and boots, what is the probability Alex likes Rock music vs. Classical?" You calculate probabilities based on how often leather jackets co-occur with Rock lovers in your memory bank.

3. **The Geometric Highway Strategy (Support Vector Machine - SVM)**: You imagine drawing a physical line down the center of the party room separating Rock fans on the left from Classical fans on the right. But you don't just draw *any* line—you push the two groups as far apart as possible, creating a wide empty highway between them. The people standing right on the edge of the highway are your **Support Vectors**.

```
KNN:          Find closest K neighbors in space -> Take majority vote.
Naive Bayes:  Multiply conditional probabilities -> Pick highest probability class.
SVM:          Build maximum width spatial highway (margin) between classes.
```

---

## 2. Core Mechanics & Granular Step-by-Step Breakdown

### 1. K-Nearest Neighbors (KNN): Lazy Database Lookups

KNN is called a **Lazy Learner** because it does ZERO work during training! It literally just saves your training data into memory. 

When a new query point $x_q$ arrives at inference time:
1. It computes spatial distance $d(x_q, x_i)$ to every single point $x_i$ in the dataset.
2. It sorts those distances ascending and picks the top $K$ closest neighbors.
3. For classification, it takes a majority vote. For regression, it averages their values.

```
       Feature 2 ^
                 │      o (Class 1)
                 │    o   * (New Query Point x_q)
                 │      x   x (Class 0)
                 │        x
                 └─────────────────────────► Feature 1
           If K=3: 2 'o's vs 1 'x' -> Predict Class 1
```

- **The Danger of K**: 
  - If $K=1$, the decision boundary wiggles around every single noisy outlier (Extreme Overfitting / High Variance).
  - If $K=100$, the boundary becomes a blurry blob, ignoring local structure (Underfitting / High Bias).

---

### 2. Naive Bayes: The Probabilistic Frequency Counter

Naive Bayes uses **Bayes' Theorem** to calculate posterior class probabilities $P(Y=c \mid X)$.

Imagine you are building a spam filter inspecting an incoming email with words: `"WINNER", "FREE", "BITCOIN"`.
You want to calculate:

$$P(\text{Spam} \mid \text{"WINNER", "FREE", "BITCOIN"})$$

To compute this easily, Naive Bayes makes a bold, simplifying assumption: **it assumes that features are completely independent of each other given the class.** 

That means it assumes the word `"FREE"` appearing in an email has zero correlation with the word `"WINNER"` appearing. Is this assumption strictly true in real life? Of course not! Words appear together in phrases all the time. But making this **"Naive"** assumption allows us to multiply simple independent probabilities together, creating a model that runs at lightning speed!

#### Laplace Smoothing (Preventing Zero-Probability Disasters)
What if the word `"Bitcoin"` never appeared in any legitimate non-spam emails in your training dataset?
Then $P(\text{"Bitcoin"} \mid \text{Ham}) = 0$. 

Because Naive Bayes multiplies probabilities together:

$$P(\text{"WINNER"} \mid \text{Ham}) \times P(\text{"FREE"} \mid \text{Ham}) \times P(\text{"Bitcoin"} \mid \text{Ham}) = \text{anything} \times \text{anything} \times 0 = 0!$$

A single unseen word causes the entire probability to collapse to absolute zero! To fix this, we apply **Laplace Smoothing**: adding $+1$ to every word count numerator so no probability is ever zero.

---

### 3. Support Vector Machines (SVM): The Maximum Margin Highway

Imagine two armies facing each other on a battlefield. You want to erect a defensive border wall between them. Where do you place the wall? 

You place it right in the middle of the empty buffer zone, pushing the wall as far away as possible from the front-line soldiers of both armies!

```
        Class +1              Support Vectors (Front-line soldiers)
           o      o                   │
         o    o   o ──┐               ▼
──────────────────────┼───────────── (Hyperplane: w^T x + b = +1)
          .  /  .     │     Margin
  ========.=/===.===================  (Optimal Decision Boundary: w^T x + b = 0)
          ./  /  .    │     Margin
──────────────────────┼───────────── (Hyperplane: w^T x + b = -1)
              x   x   └───► ▲
            x   x           │
        Class -1       Support Vectors
```

- **Decision Hyperplane**: $w^T x + b = 0$.
- **Support Vectors**: The data points landing right on the margins ($w^T x + b = \pm 1$). *Crucial Insight*: If you delete 90% of all other data points far away from the border, the learned SVM decision boundary **remains 100% identical!** Only the front-line support vectors matter.
- **The Kernel Trick (Bending Space)**: What if data cannot be separated by a straight line in 2D? (e.g., Red points form a circle inside a ring of Blue points).
  - *The Magic*: You project 2D data into a higher 3D space using a **Kernel Function** $K(x_i, x_j)$ (like lifting the red center points upward into a 3D bowl shape). In 3D space, a flat sheet of paper can cleanly slice between red and blue points!

---

## 3. Visual Architecture: The Kernel Trick Demonstration

```
2D Non-Separable Space (Red circle inside Blue ring):
       y ^    blue  blue
         │  blue  red  blue
         │    blue  blue
         └──────────────────► x  (Cannot separate with straight line!)

3D Transformation via RBF Kernel (z = x^2 + y^2):
       z ^
         │        blue  blue  blue  (High z)
         │  ----------------------  (3D Flat Plane Slices Separating Layer!)
         │        red   red   red   (Low z)
         └──────────────────────────► x, y
```

---

## 4. Practical Implementation: Benchmarking Models

Let's write a complete script using `scikit-learn` to benchmark KNN, Naive Bayes, and SVM on a medical classification dataset:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# 1. Load Data & Split
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# CRITICAL: Distance-based models (KNN, SVM) MANDATORILY require feature scaling!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 1. K-Nearest Neighbors ---
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

# --- 2. Naive Bayes (Gaussian) ---
nb = GaussianNB()
nb.fit(X_train, y_train)  # NB does not strictly require scaling
y_pred_nb = nb.predict(X_test)

# --- 3. Support Vector Machine (RBF Kernel) ---
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)

# Display Comparative Benchmark Metrics
print("=== CLASSIFIER BENCHMARK RESULTS ===")
print(f"KNN (K=5)     | Accuracy: {accuracy_score(y_test, y_pred_knn):.4f} | F1-Score: {f1_score(y_test, y_pred_knn):.4f}")
print(f"Naive Bayes   | Accuracy: {accuracy_score(y_test, y_pred_nb):.4f} | F1-Score: {f1_score(y_test, y_pred_nb):.4f}")
print(f"SVM (RBF)     | Accuracy: {accuracy_score(y_test, y_pred_svm):.4f} | F1-Score: {f1_score(y_test, y_pred_svm):.4f}")
```

---

## 5. Demystifying the Mathematics: Explaining Every Formula

Let's unpack the mathematical machinery behind these algorithms.

### Equation 1: Bayes' Theorem with Independence Assumption
$$
P(Y=c \mid x_1, \dots, x_d) = \frac{P(Y=c) \prod_{i=1}^{d} P(x_i \mid Y=c)}{P(x_1, \dots, x_d)}
$$

- $P(Y=c \mid X)$: The **Posterior Probability** (the probability that an email is Spam given its words).
- $P(Y=c)$: The **Prior Probability** (the base rate frequency of Spam in your training set).
- $\prod_{i=1}^d P(x_i \mid Y=c)$: The **Likelihood**. The product $\prod$ multiplies individual conditional probabilities $P(x_1|c) \times P(x_2|c) \dots$ together under the independence assumption.
- $P(x_1, \dots, x_d)$: The **Evidence** denominator scaling probabilities so they sum to 1.0.

---

### Equation 2: Laplace Smoothing Formula
$$
P(x_i \mid Y=c) = \frac{\text{count}(x_i, c) + \alpha}{\sum_{x} \text{count}(x, c) + \alpha \cdot d}
$$

- $\text{count}(x_i, c)$: Number of times word $x_i$ appears in class $c$ documents.
- $\alpha$ (alpha): The smoothing parameter (usually $\alpha = 1$, called **Add-One / Laplace Smoothing**).
- Adding $+ \alpha$ to the top and $+ \alpha \cdot d$ to the bottom guarantees that if count is $0$, the probability becomes $\frac{1}{\text{total} + d} > 0$, preventing zero-multiplication crashes!

---

### Equation 3: The Radial Basis Function (RBF) Kernel
$$
K(x_i, x_j) = \exp\left( -\gamma \|x_i - x_j\|^2 \right)
$$

- $\|x_i - x_j\|^2$: Squared Euclidean distance between vector $x_i$ and vector $x_j$.
- $\gamma$ (gamma): Controls the radius of influence of individual support vectors:
  - Small $\gamma$: Large bell-curve radius $\to$ smooth, generalized boundaries.
  - Large $\gamma$: Small pin-point radius $\to$ complex, wiggly boundaries (risk of overfitting).
- $\exp(-\dots)$: If distance is $0$, $e^{0} = 1.0$ (maximum similarity). As distance grows huge, $e^{-\text{huge}} \to 0.0$.

---

## 6. Real-World Production Gotchas & Failure Modes

### 1. KNN Inference Latency ($O(n \cdot d)$ Memory Wall)
Because KNN scans the ENTIRE training dataset for every single prediction, its query time on 10 million rows can exceed 10 seconds!
- *Fix*: Never use raw KNN in production for large datasets. Use Approximate Nearest Neighbors libraries (**FAISS** or **HNSW**).

### 2. SVM Scaling Sensitivity
If `Income` ($\$0 - \$100k$) is unscaled next to `Age` ($0 - 100$), SVM margins become elongated oval distortions. Always normalize inputs with `StandardScaler` prior to fitting SVMs!

---

## 7. Feynman Exercises & Deep Thinking Challenges

1. **Thought Experiment**: Suppose you have an SVM model with 10,000 training points, but only 20 of them end up as Support Vectors. If you delete 9,980 non-support vector points and re-train the model on the remaining 20 points, what will happen to the learned decision boundary?
   - *Answer/Explanation*: The boundary will be **100% identical!** SVM parameters $w$ and $b$ depend strictly on the boundary support vectors. Non-support vectors have dual multipliers $\alpha_i = 0$, exerting zero influence on the solution.

2. **Code Challenge**: Train an SVM with `C=0.001` versus `C=1000`. Observe how small $C$ creates a smooth underfitting boundary, while huge $C$ creates a wiggly overfitting boundary.
