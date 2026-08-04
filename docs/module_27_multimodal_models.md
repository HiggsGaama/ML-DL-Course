# Module 27: Multimodal Models (CLIP & Joint Embeddings)

## 1. The Physical Intuition: Bridging Two Worlds

Imagine you have two friends: one is blind but understands language perfectly (Text Encoder), and the other is deaf but has perfect 20/20 visual vision (Vision Encoder).

How do you get them to agree on what a photograph depicts?

You build a **Joint Multimodal Embedding Space** ($\mathbb{R}^d$). 

You ask your visual friend to look at an image of a dog and output a 512-dimensional vector. You ask your blind friend to read the text description *"A golden retriever playing fetch"* and output a 512-dimensional vector.

Then, you adjust their internal neural networks until the **text vector and the image vector sit at the exact same geometric coordinates in space!**

```
Image X ──► [ Vision Encoder ViT ] ──► Image Embedding I ∈ ℝ^d ──┐
                                                                 ├──► Cosine Similarity Matrix I * T^T
Text  Y ──► [ Text Encoder Transformer ] ──► Text Embedding T ∈ ℝ^d ──┘
```

This is **CLIP (Contrastive Language-Image Pre-training)**, created by OpenAI in 2021. CLIP powers zero-shot image classification, text-guided image generation, and multimodal visual search engines!

---

## 2. Core Concepts & Granular Step-by-Step Breakdown

### 1. Contrastive Learning via Symmetric Cross-Entropy

CLIP is pre-trained on 400 million (Image, Text) pairs collected from the internet.

For a mini-batch of $N$ images and $N$ texts:
- It computes an $N \times N$ similarity matrix of dot products between all image vectors $I_1 \dots I_N$ and text vectors $T_1 \dots T_N$.
- The diagonal entries $(I_i, T_i)$ represent true matching pairs. The off-diagonal entries represent non-matching negative pairs.
- CLIP optimizes a **Symmetric Cross-Entropy Loss** forcing diagonal values to $1.0$ and off-diagonal values to $0.0$!

---

### 2. Zero-Shot Image Classification
How do you classify images using CLIP without training a new classifier?
1. Convert candidate class names into text prompts: `["A photo of a dog", "A photo of a cat", "A photo of a car"]`.
2. Pass prompts through the CLIP Text Encoder to get class vectors $T_1, T_2, T_3$.
3. Pass a query image through CLIP Vision Encoder to get image vector $I$.
4. Compute cosine similarity between $I$ and all class vectors $T_k$. The highest similarity score wins!

---

## 3. Practical Implementation: Zero-Shot Classification with CLIP

Let's write a complete Python script demonstrating CLIP zero-shot image classification using Hugging Face `transformers`:

```python
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 1. Load Pre-trained CLIP Model & Processor
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

# 2. Synthetic Test Inputs
# Define candidate zero-shot text labels
candidate_labels = ["a photo of a sports car", "a photo of a golden retriever", "a photo of a laptop computer"]

# Create dummy RGB image (300x300 pixels)
dummy_image = Image.new('RGB', (300, 300), color=(255, 100, 50))

# 3. Preprocess Inputs
inputs = processor(
    text=candidate_labels,
    images=dummy_image,
    return_tensors="pt",
    padding=True
)

# 4. Execute Multimodal Forward Pass
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # Image-to-text similarity logits
    probs = logits_per_image.softmax(dim=1)     # Convert to probabilities

print("=== CLIP ZERO-SHOT CLASSIFICATION ===")
for label, prob in zip(candidate_labels, probs[0].tolist()):
    print(f"  {label:<30}: {prob*100:.2f}%")
```

---

## 4. Demystifying the Mathematics: Explaining Every Formula

Let's dissect the Symmetric InfoNCE Loss driving CLIP.

### Equation 1: Symmetric InfoNCE Loss
$$
\mathcal{L}_{\text{InfoNCE}} = \frac{\mathcal{L}_{I \to T} + \mathcal{L}_{T \to I}}{2}
$$

$$\mathcal{L}_{I \to T} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp\left( \text{sim}(I_i, T_i) / \tau \right)}{\sum_{j=1}^{N} \exp\left( \text{sim}(I_i, T_j) / \tau \right)}$$

- $\text{sim}(I_i, T_j) = \frac{I_i \cdot T_j}{\|I_i\| \|T_j\|}$: Cosine similarity between image vector $I_i$ and text vector $T_j$.
- $\tau$ (tau): Learnable temperature parameter scaling dot products.
- Maximizes diagonal pair alignment while minimizing cross-pair similarity!

---

## 5. Real-World Production Gotchas & Failure Modes

### Prompt Engineering Sensitivity
Zero-shot CLIP classification is sensitive to prompt syntax. Passing raw class names like `"Dog"` yields lower accuracy than passing wrapped context templates like `"A high quality photo of a Dog."`

---

## 6. Feynman Exercises & Deep Thinking Challenges

1. **Thought Experiment**: Why does CLIP generalize so well to unseen datasets compared to ImageNet classifiers?
   - *Answer/Explanation*: ImageNet models are hardcoded to 1,000 fixed class indices. CLIP learns open-vocabulary semantic concepts across 400M natural language text descriptions.

2. **Exercise**: Extract image vectors $I$ and text vectors $T$ using CLIP in PyTorch and compute their dot product similarity manually.
