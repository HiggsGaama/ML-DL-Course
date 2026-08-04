# Module 21: Word Embeddings & Vector Spaces

## 1. The Physical Intuition: Mapping Meaning into Geometry

Imagine you want to teach a computer the meaning of words: `"King"`, `"Queen"`, `"Apple"`, `"Orange"`, `"Man"`, `"Woman"`.

If you use One-Hot Encoding (`"King": [1, 0, 0, 0, 0]`, `"Queen": [0, 1, 0, 0, 0]`), the spatial distance between EVERY pair of words is identical! The dot product between `"King"` and `"Queen"` is $0.0$, and the dot product between `"King"` and `"Apple"` is also $0.0$. The computer assumes `"King"` is no more similar to `"Queen"` than it is to a piece of fruit!

**Word Embeddings (Word2Vec)** project words into a continuous high-dimensional geometric vector space ($\mathbb{R}^d$, typically $d=300$). 

Words with similar semantic meanings cluster together in space. Even more amazing, semantic relationships become **vector arithmetic equations**:

$$\vec{v}_{\text{King}} - \vec{v}_{\text{Man}} + \vec{v}_{\text{Woman}} \approx \vec{v}_{\text{Queen}}$$

```
Semantic Vector Space:
           Gender Axis ^
                       │       Queen  •
                       │             /
                       │            /  (Gender Vector Shift)
                       │   King  • /
                       │
                       │       Woman  •
                       │             /
                       │   Man   • /
                       └─────────────────────────────► Royalty Axis
```

---

## 2. Core Concepts & Granular Step-by-Step Breakdown

### 1. Word2Vec Architectures

- **Skip-gram**: Given a center target word (e.g., `"rocket"`), predict its surrounding context words (`["launched", "into", "orbit"]`).
- **CBOW (Continuous Bag of Words)**: Given context words, predict the missing center word.
- **Negative Sampling Loss**: Instead of computing a computationally expensive Softmax across all 500,000 words in the vocabulary, binary logistic regression is performed on 1 true target word + $K$ random negative noise words!

---

### 2. Subword Embeddings (FastText)
Standard Word2Vec fails on Out-Of-Vocabulary (OOV) words. **FastText** breaks words down into character $n$-grams (e.g., `"apple"` $\to$ `["<ap", "app", "ppl", "ple", "le>"]`). It constructs embeddings by summing subword n-gram vectors, allowing it to generate embeddings for unseen words like `"appletunish"`!

---

## 3. Practical Implementation: Word Embeddings in PyTorch

Let's write a complete PyTorch script demonstrating an `nn.Embedding` layer lookup table and vector similarity calculations:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Vocabulary Mapping
vocab = {"king": 0, "man": 1, "woman": 2, "queen": 3, "apple": 4}
vocab_size = len(vocab)
embedding_dim = 4

# 2. Instantiate PyTorch Embedding Layer (Lookup Table: Vocab_Size x Embedding_Dim)
torch.manual_seed(42)
embeddings = nn.Embedding(vocab_size, embedding_dim)

# Extract embedding vectors
v_king  = embeddings(torch.tensor(vocab["king"]))
v_man   = embeddings(torch.tensor(vocab["man"]))
v_woman = embeddings(torch.tensor(vocab["woman"]))
v_queen = embeddings(torch.tensor(vocab["queen"]))

# Compute Vector Arithmetic: King - Man + Woman
v_result = v_king - v_man + v_woman

# Compute Cosine Similarity between Result Vector and Queen Vector
cos_sim = F.cosine_similarity(v_result.unsqueeze(0), v_queen.unsqueeze(0))

print("=== PYTORCH EMBEDDING DEMO ===")
print(f"Vector for 'King':   {v_king.detach().numpy().round(3)}")
print(f"Calculated Result:   {v_result.detach().numpy().round(3)}")
print(f"Vector for 'Queen':  {v_queen.detach().numpy().round(3)}")
print(f"Cosine Similarity (Result vs Queen): {cos_sim.item():.4f}")
```

---

## 4. Demystifying the Mathematics: Explaining Every Formula

Let's dissect the Word2Vec Negative Sampling Loss function.

### Equation 1: Negative Sampling Loss
$$
\mathcal{L}_{\text{NegSample}} = -\log \sigma(v_{w_O}'^T v_{w_I}) - \sum_{k=1}^{K} \log \sigma(-v_{w_k}'^T v_{w_I})
$$

- $w_I$: Input center word vector.
- $w_O$: True output context word vector.
- $w_k$: $K$ random negative noise words drawn from unigram distribution $P_n(w)$.
- $\sigma(v_{w_O}'^T v_{w_I})$: Maximize probability that true context word $w_O$ co-occurs with center word $w_I$.
- $\sigma(-v_{w_k}'^T v_{w_I})$: Maximize probability that random noise words $w_k$ DO NOT co-occur with center word $w_I$.

---

## 5. Real-World Production Gotchas & Failure Modes

### Static vs. Contextual Embeddings
Word2Vec produces **static embeddings**: the word `"bank"` has the exact same static vector whether used in *"river bank"* or *"investment bank"*. Modern Transformers (BERT) produce **contextual embeddings** dynamically adjusting vectors based on surrounding text!

---

## 6. Feynman Exercises & Deep Thinking Challenges

1. **Thought Experiment**: Why does Word2Vec assign similar vectors to `"cat"` and `"dog"` even though they are different animals?
   - *Answer/Explanation*: Both words appear in identical sentence contexts (*"The ___ ate its food"*, *"The ___ slept on the rug"*), forcing SGD optimization to push their vectors to the same spatial region.

2. **Exercise**: Load pre-trained GloVe vectors using `gensim` and find the top 5 most similar words to `"python"`.
