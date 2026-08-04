# Module 11: Anomaly Detection

## 1. The Physical Intuition: Spotting the Black Swan

Imagine you are standing at a busy pedestrian crosswalk in downtown Tokyo. Thousands of people cross the street every hour: salarymen in suits, students in uniforms, tourists with backpacks. Your brain smoothly registers all of them as "normal baseline patterns".

Suddenly, a person wearing a neon-green astronaut suit riding a unicycle rolls into the crosswalk. 

Your eyes lock onto them instantly. Why? You didn't receive a formal lecture beforehand on "astronaut unicyclists". Your brain automatically constructed a continuous spatial model of normal pedestrian attributes, and the astronaut instantly fell way outside the density boundary of that space!

```
Normal Data Stream ──► [ Model Learns Boundary Space ] ──► Point Inside Boundary (Normal)
                                                      └──► Point Outside Boundary (ANOMALY!)
```

In software engineering, **Anomaly Detection** is automated outlier detection without explicit anomaly labels. Instead of hardcoding fragile static threshold alert rules (`CPU > 95%`), anomaly detection algorithms learn the high-dimensional spatial boundaries of normal behavior and flag any data point landing far outside those boundaries.

---

## 2. Core Concepts & Granular Step-by-Step Breakdown

### 1. Statistical Methods (Z-Score / IQR)
- Measures how many standard deviations ($\sigma$) a data point lies away from the distribution mean ($\mu$).
- *Limitation*: Assumes univariate (1D) Gaussian distribution shape. Fails on complex multi-dimensional data.

---

### 2. Isolation Forest: The Fast Tree-Chopping Trick

Most machine learning algorithms try to model *normal* data points. **Isolation Forest** flips the problem upside down: it explicitly focuses on isolating outliers!

```
NORMAL DENSE CLUSTER (Requires many random cuts):
  • • • • • • •
  • • • • • • •  <-- Cut 1, Cut 2, Cut 3, Cut 4, Cut 5... (Hard to isolate 1 point)
  • • • • • • •

ANOMALOUS OUTLIER (Isolated in 1 random cut):
                                   *  <-- Cut 1 (Boom! Isolated immediately!)
```

- *The Insight*: Outliers are few and structurally different. Therefore, if you build random decision trees using random split thresholds, **anomalous points require far FEWER random splits to isolate into single leaf nodes** than normal dense cluster points!
- An anomaly score $s(x, n)$ is calculated based on the average path length $\mathbb{E}(h(x))$ across the forest. Short path $\to$ Anomaly!

---

### 3. One-Class SVM: Enclosing Normal Space

Fits a tight non-linear boundary sphere surrounding normal data points in high-dimensional kernel space. Any test point landing outside this boundary sphere is flagged as an anomaly.

---

## 3. Visual Architecture: Isolation Forest Path Length

```
Normal Dense Sample (Average Path Length E(h) = 12 splits)
  Root ──► Node 1 ──► Node 2 ──► Node 3 ──► ... ──► Leaf (Deep in tree)

Anomalous Outlier Sample (Average Path Length E(h) = 2 splits)
  Root ──► Node 1 ──► Leaf (Isolated almost instantly!)
```

---

## 4. Practical Implementation: Detecting Outliers

Let's write a complete script using `scikit-learn` demonstrating Isolation Forest, One-Class SVM, and Elliptic Envelope:

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

# 1. Synthesize 300 normal samples + 20 extreme outliers
X_normal, _ = make_blobs(n_samples=300, centers=1, cluster_std=1.0, random_state=42)
rng = np.random.RandomState(42)
X_outliers = rng.uniform(low=-10, high=10, size=(20, 2))
X = np.vstack([X_normal, X_outliers])

# --- Method 1: Isolation Forest ---
iso = IsolationForest(contamination=0.06, random_state=42)
preds_iso = iso.fit_predict(X) # 1 for Normal, -1 for Anomaly
print(f"Isolation Forest Flagged Anomalies Count: {sum(preds_iso == -1)}")

# --- Method 2: One-Class SVM ---
oc_svm = OneClassSVM(nu=0.06, kernel='rbf', gamma='scale')
preds_svm = oc_svm.fit_predict(X)
print(f"One-Class SVM Flagged Anomalies Count:    {sum(preds_svm == -1)}")

# --- Method 3: Elliptic Envelope ---
ee = EllipticEnvelope(contamination=0.06, random_state=42)
preds_ee = ee.fit_predict(X)
print(f"Elliptic Envelope Flagged Anomalies Count: {sum(preds_ee == -1)}")
```

---

## 5. Demystifying the Mathematics: Explaining Every Formula

Let's unpack the mathematical equation behind the Isolation Forest Anomaly Score.

### Equation 1: Isolation Forest Anomaly Score
$$
s(x, n) = 2^{-\frac{\mathbb{E}(h(x))}{c(n)}}
$$

- $x$: The candidate sample being evaluated.
- $h(x)$: Path length (number of splits) required to isolate sample $x$ in a decision tree.
- $\mathbb{E}(h(x))$: Average path length of sample $x$ across all trees in the forest.
- $c(n)$: Average path length of unsuccessful searches in a binary search tree built on $n$ nodes.
- Look at how the exponent behaves:
  - If $\mathbb{E}(h(x)) \to 0$ (isolated almost instantly), $\frac{\mathbb{E}(h(x))}{c(n)} \to 0 \to 2^0 = 1.0$ (**Confirmed Anomaly!**).
  - If $\mathbb{E}(h(x)) \to c(n)$ (normal average path), $2^{-1} = 0.5$ (**Normal Data Point**).

---

## 6. Real-World Production Gotchas & Failure Modes

### Concept Drift & Shifted Baselines
Normal operational behavior changes over time (e.g., Black Friday traffic volume is $10\times$ normal traffic). Re-fit anomaly models periodically to update normal baseline bounds.

---

## 7. Feynman Exercises & Deep Thinking Challenges

1. **Thought Experiment**: Why does Isolation Forest perform significantly better than Z-score on multi-modal distributions (e.g., two separate normal clusters)?
   - *Answer/Explanation*: Z-score assumes a single Gaussian distribution around a central mean, flagging points between the two clusters as normal. Isolation Forest makes zero parametric distribution assumptions, easily isolating points in sparse gaps.

2. **Exercise**: Compare execution speed and memory consumption of Isolation Forest vs. One-Class SVM on a 50,000 row dataset.
