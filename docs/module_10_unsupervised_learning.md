# Module 10: Unsupervised Learning & Dimensionality Reduction

## 1. The Physical Intuition: Finding Galaxies and Shadow Projections

Imagine you are an astronomer looking through a telescope at a dark night sky filled with 500,000 scattered stars. You have no labels ($y$). No teacher told you which stars belong to which solar systems or galaxies.

Yet, as you stare at the sky, your eyes naturally discover clusters: *"Look! That dense cloud of stars over there is a spiral galaxy. That separate cluster over here is a globular cluster."*

In Machine Learning:
- **Clustering (K-Means / DBSCAN)**: Automatically partitions unlabelled data matrices into discrete groups based on spatial density and proximity.
- **Dimensionality Reduction (PCA / t-SNE)**: Compresses a high-dimensional feature space ($d=500$) down to 2 or 3 principal axes. Think of holding a complex 3D object in front of a flashlight—its 2D shadow on the wall captures the essential outline shape of the object!

```
High Dimensional Data (500 columns) ──► [ PCA Projection ] ──► Compressed 2D Points (Visualization)
Unlabelled Vector Space             ──► [ K-Means ]        ──► Auto-Discovered Cluster Labels [0, 1, 2]
```

---

## 2. Core Concepts & Granular Step-by-Step Breakdown

### 1. K-Means Clustering: The Dancing Centroids

K-Means is a mechanical iterative dance between centroids and data points:

```
Step 1: Random Centroids     Step 2: Assign Points to Centroids   Step 3: Shift Centroids to Means
      x     *                       x     *                              x  *
    .   .   .  *                  .   .   .  *                         .   .  *
      *  x                          *  x                                 * x
 (Initial Guesses)              (Voronoi Distance Regions)         (Centroids Move to Center)
```

1. You pick $K$ (e.g., $K=4$) and place $K$ random centroid markers in feature space.
2. **Assignment Step**: Every data point computes its Euclidean distance to all $K$ centroids and assigns itself to the closest centroid.
3. **Update Step**: Each centroid recalculates its position by taking the exact mean average of all data points assigned to it, sliding to the center of its cluster.
4. Repeat steps 2 and 3 until centroids stop moving.

---

### 2. DBSCAN: Density-Based Spatial Clustering

What if your clusters aren't neat spherical blobs? What if they form complex spiral arms or concentric rings? K-Means fails completely on non-spherical shapes!

**DBSCAN** works differently: it grows clusters by stepping along dense chains of points (points within distance $\epsilon$ containing at least `min_samples` neighbors). Any sparse point standing alone in empty space is automatically flagged as **Noise / Outlier (`-1`)**!

---

### 3. Principal Component Analysis (PCA): Rotating Space for Maximum Variance

PCA finds new coordinate axes (Principal Components) that capture the maximum spread (variance) of your data.

```
Original Feature Axes (x1, x2):                 PCA Rotated Principal Axes (PC1, PC2):
x2 ^       .  .  *                              PC2 ^
   │    .  *  .                                     │    .  .  *  .  .
   │ .  .  *                                        └─────────────────────────► PC1
   └──────────────────► x1                            (PC1 captures maximum variance!)
```

- **PC1**: The single line through space along which the data is stretched out the most.
- **PC2**: The line orthogonal (perpendicular) to PC1 that captures the second-most variance.

---

## 3. Visual Architecture: PCA Eigendecomposition Pipeline

```
Zero-Centered Matrix X ──► [ Compute Covariance Matrix Σ ] ──► [ Eigendecomposition ] ──► Eigenvalues λ (Variance)
                                                                                       └──► Eigenvectors v (Directions)
```

---

## 4. Practical Implementation: Clustering and Dimensionality Reduction

Let's write a complete script using `scikit-learn` demonstrating K-Means, DBSCAN, PCA, and t-SNE:

```python
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# 1. Synthesize benchmark cluster data
X_blobs, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=42)

# --- K-Means Clustering ---
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels_km = kmeans.fit_predict(X_blobs)
print(f"K-Means Silhouette Score: {silhouette_score(X_blobs, labels_km):.4f}")

# --- DBSCAN Clustering (Non-Spherical Moons Data) ---
X_moons, _ = make_moons(n_samples=500, noise=0.05, random_state=42)
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels_db = dbscan.fit_predict(X_moons)
print(f"DBSCAN Cluster Labels Found: {np.unique(labels_db)} (-1 represents noise)")

# --- PCA Dimensionality Reduction ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_blobs)
print(f"PCA Preserved Variance Ratios: {pca.explained_variance_ratio_} (Total: {sum(pca.explained_variance_ratio_):.4f})")

# --- t-SNE Visualization Embedding ---
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_blobs)
print(f"t-SNE Output Matrix Shape: {X_tsne.shape}")
```

---

## 5. Demystifying the Mathematics: Explaining Every Formula

Let's unpack the equations driving PCA and clustering evaluation.

### Equation 1: PCA Covariance Matrix & Eigendecomposition
Given a zero-centered feature matrix $X \in \mathbb{R}^{n \times d}$:

$$\Sigma = \frac{1}{n} X^T X$$

$$\Sigma v_k = \lambda_k v_k$$

- $\Sigma$ (sigma): The $d \times d$ **Covariance Matrix**. Entry $\Sigma_{i,j}$ measures how feature $i$ and feature $j$ vary together.
- $v_k$: The **Eigenvectors** (Principal Component direction vectors). They define the rotated axes.
- $\lambda_k$ (lambda): The **Eigenvalues**. They measure how much variance (stretch) exists along each eigenvector direction $v_k$!

---

### Equation 2: Silhouette Coefficient Formula
How do we mathematically judge whether a cluster configuration is good?

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

- $a(i)$: Mean distance from sample $i$ to all other points in the *same* cluster (intra-cluster distance). We want this small!
- $b(i)$: Mean distance from sample $i$ to all points in the *nearest neighboring* cluster (inter-cluster distance). We want this large!
- $s(i) \to +1.0$: Sample $i$ is well inside its cluster.
- $s(i) \to -1.0$: Sample $i$ is assigned to the wrong cluster!

---

## 6. Real-World Production Gotchas & Failure Modes

### Distortion Warning in t-SNE
Distances between distant clusters in t-SNE plots are mathematically meaningless! Do NOT use t-SNE output vectors as inputs to downstream distance metrics; use PCA vectors instead.

---

## 7. Feynman Exercises & Deep Thinking Challenges

1. **Thought Experiment**: Why must features be standardized with `StandardScaler` prior to running PCA?
   - *Answer/Explanation*: PCA maximizes variance. Unscaled features with large raw values (e.g., Salary) will falsely dominate principal component direction vectors regardless of true information content.

2. **Code Challenge**: Plot Silhouette Scores for $K \in [2, 10]$ on a dataset to locate the optimal number of clusters using the Elbow method.
