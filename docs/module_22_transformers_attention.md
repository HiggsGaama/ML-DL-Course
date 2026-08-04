# Module 22: Transformers & Self-Attention Mechanics

## 1. The Physical Intuition: The Self-Attention Revolution

Imagine you are attending a cocktail party with 50 domain experts. You hold a specific question in your mind: *"How do I fix a distributed database deadlock?"*

Do you listen to all 50 experts talking simultaneously with equal attention? No!

You hold up your question (**Query $Q$**). You scan the name badges (**Keys $K$**) of everyone in the room. You calculate a matching score between your question and each person's badge. 

The person wearing a badge *"Database Architect"* gets a 99% match score! You focus your attention almost exclusively on them, listening to their knowledge (**Value $V$**), while ignoring the chatter of the wedding planner.

```
RNN Sequential Processing:      Word 1 ──► Word 2 ──► Word 3 ──► Word 4 (Slow, linear bottleneck)
Transformer Self-Attention:     [ All Words Interconnect Simultaneously via Q, K, V Matrices! ]
```

In 2017, Vaswani et al. published *"Attention Is All You Need"*, launching the Transformer revolution. Transformers discarded sequential RNN loops entirely, allowing every word in a sequence to look at every other word **simultaneously in parallel** via **Scaled Dot-Product Attention**!

---

## 2. Core Concepts & Granular Step-by-Step Breakdown

### 1. The Query, Key, and Value ($Q, K, V$) Abstraction

For every word input vector $X$:
1. We project $X$ through three learnable weight matrices ($W_Q, W_K, W_V$) to create:
   - **Query ($Q$)**: What this word is searching for.
   - **Key ($K$)**: What information this word contains (its indexing badge).
   - **Value ($V$)**: The actual content payload of the word.

2. **Dot-Product Attention Score**:
   - Taking the dot product $Q K^T$ calculates raw similarity scores between every pair of words.
   - We divide by $\sqrt{d_k}$ to prevent dot products from growing massive in high dimensions (which would push Softmax into regions with vanishing gradients).
   - We pass scores through **Softmax** to convert them into attention weights summing to 1.0.
   - We multiply attention weights by **Value $V$** to construct the updated contextual representation!

$$\text{Attention}(Q, K, V) = \text{Softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$$

---

### 2. Multi-Head Attention
Instead of computing attention once, **Multi-Head Attention** splits queries, keys, and values into $h$ parallel heads (e.g., $h=8$). One head focuses on syntactic grammar relationships; another head focuses on pronoun references; another on semantic entities!

---

## 3. Architecture & Visual Diagrams

### Transformer Multi-Head Self-Attention Pipeline

```
Input Embeddings + Positional Encodings
              │
              ├──► Query Matrix Q  = X * W_Q
              ├──► Key Matrix K    = X * W_K
              └──► Value Matrix V  = X * W_V
                        │
                        ▼
   [ Scaled Dot Product: (Q * K^T) / sqrt(d_k) ]
                        │
                        ▼
            [ Softmax Attention Weights ]
                        │
                        ▼
     [ Attention Weights * Value Matrix V ] ──► Multi-Head Concatenation ──► Output
```

---

## 4. Practical Implementation: Multi-Head Attention in PyTorch

Let's write a pure PyTorch script demonstrating Scaled Dot-Product Attention and `nn.MultiheadAttention`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Manual Scaled Dot-Product Attention Function
def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)
    # Step 1: Compute raw similarity scores Q * K^T
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    # Step 2: Softmax to get normalized attention probability weights
    attn_weights = F.softmax(scores, dim=-1)
    # Step 3: Multiply by Value payload matrix V
    output = torch.matmul(attn_weights, V)
    return output, attn_weights

# 2. Test Execution
seq_len = 4
d_model = 8

Q = torch.randn(1, seq_len, d_model)
K = torch.randn(1, seq_len, d_model)
V = torch.randn(1, seq_len, d_model)

context_vecs, weights = scaled_dot_product_attention(Q, K, V)

print("=== SCALED DOT-PRODUCT ATTENTION ===")
print("Attention Weights Matrix Shape:", weights.shape)
print("Attention Weights (Row sum must equal 1.0):\n", weights.detach().numpy().round(4))
print("Output Context Vectors Shape:  ", context_vecs.shape)

# 3. Built-in PyTorch MultiheadAttention Module
mha = nn.MultiheadAttention(embed_dim=16, num_heads=2, batch_first=True)
dummy_x = torch.randn(2, 5, 16) # Batch=2, Seq=5, Embed=16
attn_output, _ = mha(dummy_x, dummy_x, dummy_x)
print("\nPyTorch Built-in MHA Output Shape:", attn_output.shape)
```

---

## 5. Demystifying the Mathematics: Explaining Every Formula

Let's dissect the scaling factor $\frac{1}{\sqrt{d_k}}$ in the attention equation.

### Equation 1: Scaled Dot-Product Attention
$$
\text{Attention}(Q, K, V) = \text{Softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V
$$

- $Q K^T$: Dot product between $N \times d_k$ queries and $d_k \times N$ keys, yielding an $N \times N$ similarity matrix.
- Why divide by $\sqrt{d_k}$?
  - Assume components of $q$ and $k$ are independent random variables with mean $0$ and variance $1$. Their dot product $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ has mean $0$ and variance $d_k$.
  - For large $d_k$ (e.g., $d_k = 64$), variance is $64$, pushing values up to $\pm 8$.
  - Passing large inputs into Softmax causes extreme probabilities ($1.0$ vs $0.0$), driving Softmax derivatives to zero! Dividing by $\sqrt{d_k} = \sqrt{64} = 8$ scales variance back to $1.0$, preserving healthy gradient flow!

---

## 6. Real-World Production Gotchas & Failure Modes

### Positional Encodings Requirement
Self-Attention processes all words simultaneously in parallel, making it completely permutation-invariant! The sentence *"Dog bites man"* produces identical attention outputs to *"Man bites dog"*.
- *Fix*: You MUST add **Positional Encodings** (sinusoidal functions or learnable position embeddings) to input vectors before feeding them into Attention layers!

---

## 7. Feynman Exercises & Deep Thinking Challenges

1. **Thought Experiment**: What is the computational complexity of Self-Attention with sequence length $N$? Why does this make processing 100,000-token documents difficult?
   - *Answer/Explanation*: Computing $Q K^T$ requires multiplying an $N \times d$ matrix by a $d \times N$ matrix, resulting in $O(N^2 \cdot d)$ time and memory complexity. Doubling sequence length increases memory requirements by $4\times$!

2. **Exercise**: Plot sinusoidal positional encoding curves for positions $0 \dots 100$ in Python using `matplotlib`.
