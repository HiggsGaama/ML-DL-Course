# Module 23: BERT, GPT & Transformer Feature Extraction

## 1. The Physical Intuition: Encoder-Only vs. Decoder-Only Architectures

Imagine you have two world-class literary scholars analyzing a novel:

1. **The Detective Scholar (BERT - Encoder Only)**: They are handed a full page of text with a few words blacked out by ink. They read the entire page simultaneously—looking both left (past) and right (future)—to deduce what words must fill the blanks based on full bidirectional context.

2. **The Storyteller Scholar (GPT - Decoder Only)**: They sit in front of a typewriter. They are allowed to look ONLY at the words written so far to the left. Using a causal attention mask, they predict the single next word to append to the story!

```
BERT (Encoder Only):   Bi-directional Attention (Reads left & right). Perfect for Classification & Feature Extraction.
GPT (Decoder Only):   Causal Auto-regressive Attention (Reads left only). Perfect for Text Generation.
```

In software engineering, pre-trained Transformers (like BERT) serve as ultimate feature extractors, turning unstructured text strings into rich contextual embedding vectors!

---

## 2. Core Concepts & Granular Step-by-Step Breakdown

### 1. BERT Masked Language Modeling (MLM)
- During pre-training, 15% of tokens are randomly masked (`[MASK]`). BERT learns to predict the hidden words by looking at both left and right context simultaneously.

---

### 2. Sentence Pooling Strategies
How do you compress a sequence of 512 token vectors into a single vector representing the entire document?
- **`[CLS]` Token Pooling**: Uses the first output token vector (`[CLS]`).
- **Mean Pooling (Best Practice)**: Takes the element-wise mathematical mean across all valid non-padded token vectors in the sentence:

$$\vec{v}_{\text{doc}} = \frac{\sum_{i=1}^N m_i \cdot \vec{h}_i}{\sum_{i=1}^N m_i}$$

---

## 3. Practical Implementation: BERT Feature Extractor

Let's write a complete Python script using Hugging Face `transformers` and `torch` to extract document embeddings with Mean Pooling:

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# 1. Load Pre-trained BERT Model and Tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name)

# 2. Freeze BERT parameters for Feature Extraction
for param in bert_model.parameters():
    param.requires_grad = False

# 3. Input Text Sentences
sentences = [
    "The software architecture is highly performant.",
    "Database query latency is unacceptable."
]

# 4. Tokenization & Encoding
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# 5. Extract Contextual Token Vectors
with torch.no_grad():
    outputs = bert_model(**inputs)
    last_hidden_state = outputs.last_hidden_state # Shape: (Batch, Seq_Len, 768)

# 6. Apply Mean Pooling with Attention Mask
attention_mask = inputs['attention_mask'].unsqueeze(-1).float()
sum_embeddings = torch.sum(last_hidden_state * attention_mask, dim=1)
sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
mean_pooled_embeddings = sum_embeddings / sum_mask

print("=== BERT FEATURE EXTRACTION COMPLETE ===")
print("Last Hidden State Shape: ", last_hidden_state.shape)
print("Mean Pooled Document Vector Shape:", mean_pooled_embeddings.shape)
print("Sentence 1 Embedding Vector Sample:", mean_pooled_embeddings[0][:5].numpy().round(4))
```

---

## 4. Demystifying the Mathematics: Explaining Every Formula

### Plain-English Variable Walkthrough
- $H \in \mathbb{R}^{B \times N \times d}$: Hidden state tensor output from BERT ($B$=Batch, $N$=Sequence Length, $d$=768 embedding dimension).
- $M \in \{0, 1\}^{B \times N}$: Attention mask tensor ($1$ for real word tokens, $0$ for zero-padding).

---

### Formal Mathematical Formulation

#### Mean Pooling Vector Equation
$$
\vec{u} = \frac{\sum_{i=1}^N M_i \mathbf{h}_i}{\sum_{i=1}^N M_i}
$$

- $M_i \mathbf{h}_i$: Element-wise multiplication zeroes out padded token vectors so zero-padding does not distort sentence averages!

---

## 5. Real-World Production Gotchas & Failure Modes

### Sequence Truncation Limit
BERT models have a strict maximum sequence length limit ($N=512$ tokens). Passing documents longer than 512 tokens causes unhandled runtime truncation errors!
- *Fix*: Chunk long documents into 500-token windows and average their pooled embedding vectors.

---

## 6. Feynman Exercises & Deep Thinking Challenges

1. **Thought Experiment**: Why does causal masking in GPT multiply future attention key scores by $-\infty$ before computing Softmax?
   - *Answer/Explanation*: Softmax of $-\infty$ yields exact zero ($\text{Softmax}(-\infty) = 0.0$), mathematically preventing word $t$ from attending to future tokens $t+1 \dots N$.

2. **Exercise**: Calculate cosine similarity between sentence embeddings of two similar sentences versus two completely different sentences.
