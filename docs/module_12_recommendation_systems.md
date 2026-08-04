# Module 12: Recommendation Systems Basics

## 1. The Physical Intuition: Uncovering Hidden Taste Dimensions

Imagine you own a movie rental store with 10,000 films and 100,000 customers. A customer named Alice walks up to your counter. She has rented 5 movies in her life. How do you predict whether she will enjoy a movie she has never seen before, say *"Interstellar"*?

You have two fundamentally different strategies:

1. **Content-Based Filtering (Tag Matching)**: You look at the 5 movies Alice liked. They are all tagged `Genre: Sci-Fi`, `Director: Christopher Nolan`. *"Interstellar"* also has tags `Sci-Fi` and `Nolan`. So you recommend it based on item metadata match!

2. **Collaborative Filtering & Matrix Factorization (Latent Taste Vectors)**: You don't look at movie tags at all. Instead, you look at the big table of ratings across all 100,000 customers. You discover that people who rated the exact same 5 movies high as Alice also gave *"Interstellar"* 5 stars! 

Even better, you can mathematically decompose the giant sparse rating table into **Latent Dimensions** (hidden taste axes like "Action Intensity", "Emotional Depth", "Cereal Complexity") that explain why people like what they like!

```
Content-Based:       Match item metadata tags (Genres, Directors) to User Profile vector.
Collaborative:       Match user rating histories to discover similar users/items.
Matrix Factorization (SVD): Factorize sparse User-Item table into dense User Latent Vectors & Item Latent Vectors.
```

---

## 2. Core Mechanics & Granular Step-by-Step Breakdown

### 1. Matrix Factorization via Truncated SVD

In the real world, the User-Item Rating Matrix $R$ is **99% empty**. Most users have rated only 10 out of 10,000 movies.

Matrix Factorization decomposes this sparse $M \times N$ matrix $R$ into two dense, low-rank matrices:
- **User Matrix $U$** ($M \times K$): Represents each user's preference along $K$ latent taste dimensions.
- **Item Matrix $V^T$** ($K \times N$): Represents each movie's affinity along those same $K$ latent taste dimensions.

$$\text{Predicted Rating } \hat{R}_{ij} = \text{User Vector } u_i \cdot \text{Item Vector } v_j$$

Taking the dot product of Alice's latent vector $u_{\text{Alice}}$ and Interstellar's latent vector $v_{\text{Interstellar}}$ predicts her exact numerical rating!

```
User-Item Rating Matrix R (Sparse)         User Matrix U         Item Matrix V^T
       Item 1  Item 2  Item 3              Latent Features      Item 1 Item 2 Item 3
User 1 [  5      ?      1   ]   ≈  User 1 [ 0.8  0.1 ]   x  [ 5.1    0.2    1.1 ]
User 2 [  ?      4      5   ]      User 2 [ 0.2  0.9 ]      [ 0.1    4.2    4.8 ]
```

---

## 3. Practical Implementation: Building an SVD Recommender

Let's write a complete script using `scikit-learn` and `scipy` demonstrating Item-Item Collaborative Filtering and SVD Matrix Factorization:

```python
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

# 1. Create dummy User-Item rating matrix (Users x Movies)
ratings_dict = {
    'Movie_SciFi': [5, 4, 1, 0, 0],
    'Movie_Action': [4, 5, 1, 1, 0],
    'Movie_Romance': [1, 0, 5, 4, 5],
    'Movie_Drama': [0, 1, 4, 5, 4]
}
users = ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
df_ratings = pd.DataFrame(ratings_dict, index=users)

print("=== ORIGINAL SPARSE RATING MATRIX ===")
print(df_ratings)

# --- 2. Item-Item Collaborative Filtering via Cosine Similarity ---
item_sim = cosine_similarity(df_ratings.T)
df_item_sim = pd.DataFrame(item_sim, index=df_ratings.columns, columns=df_ratings.columns)
print("\n=== ITEM-ITEM SIMILARITY MATRIX ===")
print(df_item_sim.round(2))

# --- 3. Matrix Factorization via Truncated SVD ---
R = df_ratings.values.astype(float)
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

# Decompose into U, Sigma, Vt with k=2 latent taste dimensions
U, sigma, Vt = svds(R_demeaned, k=2)
sigma = np.diag(sigma)

# Reconstruct full predicted matrix
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
df_predictions = pd.DataFrame(all_user_predicted_ratings, index=users, columns=df_ratings.columns)

print("\n=== SVD RECONSTRUCTED PREDICTED RATINGS ===")
print(df_predictions.round(2))
```

---

## 4. Demystifying the Mathematics: Explaining Every Formula

Let's dissect the mathematical equations behind recommendation engines.

### Equation 1: Cosine Similarity Metric
$$
\text{Sim}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = \frac{\sum_{k} u_k v_k}{\sqrt{\sum_{k} u_k^2} \sqrt{\sum_{k} v_k^2}}
$$

- $u, v$: Vector ratings of two users or two items.
- $u \cdot v$: Dot product multiplying corresponding ratings together.
- $\|u\|_2 \|v\|_2$: Vector magnitude normalization. This ensures a user who rates everything $5$ stars has their vector length normalized so they can be compared fairly to a critical user who rates everything $2$ stars!

---

### Equation 2: Matrix Factorization Loss Function
$$
\min_{P, Q} \sum_{(i, j) \in R_{\text{obs}}} (r_{ij} - p_i^T q_j)^2 + \lambda \left( \|p_i\|_2^2 + \|q_j\|_2^2 \right)
$$

- $r_{ij}$: Actual observed rating given by user $i$ to item $j$.
- $p_i^T q_j$: Predicted rating computed as dot product of user latent vector $p_i$ and item latent vector $q_j$.
- $\lambda (\dots)$: L2 regularization penalty preventing latent vectors from growing massive and overfitting sparse training data.

---

## 5. Real-World Production Gotchas & Failure Modes

### The Cold Start Problem
A brand new user signs up. They have zero historical ratings. Collaborative filtering algorithms cannot place them in latent space, crashing predictions!
- *Fix*: Fallback to content-based filtering, trending popular items, or an onboarding genre selection survey.

---

## 6. Feynman Exercises & Deep Thinking Challenges

1. **Thought Experiment**: Why is Item-Item collaborative filtering preferred over User-User collaborative filtering in giant platforms like Amazon or Netflix?
   - *Answer/Explanation*: User counts ($>100\text{M}$) far exceed item counts ($>1\text{M}$), and item similarity vectors (e.g., how similar *"Matrix"* is to *"Inception"*) change far slower than human mood swings.

2. **Code Challenge**: Re-implement SVD predictions using `PyTorch` with Stochastic Gradient Descent (SGD) tuning latent matrices $P$ and $Q$.
