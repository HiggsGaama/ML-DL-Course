# Module 16: Convolutional Neural Networks (CNNs) & Vision Architectures

## 1. The Physical Intuition: Sliding Magnifying Glasses

Imagine you are inspecting a giant $2000 \times 2000$ pixel painting of a forest landscape. How do your eyes scan the painting to recognize a hidden tiger in the trees?

Do you look at all 4,000,000 pixels simultaneously in one single giant glance? No! Your eyes scan the canvas locally, sliding a small visual focal window across the image. 

First, your eyes spot small local features: sharp vertical lines, orange color textures, black stripe patterns. Then, higher up in your brain, those local line patterns combine into shapes: an ear, an eye, a snout. Finally, those shapes combine into a high-level concept: *"A tiger!"*

Standard MLPs fail at vision because they flatten 2D images into 1D vectors, destroying spatial relationships. 

A **Convolutional Neural Network (CNN)** mimics the biological visual cortex. It slides small 2D spatial filter windows (**Kernels**) across an image matrix, extracting local feature maps while maintaining spatial geometry!

```
Input Image (32x32x3) ──► [ Layer 1: Edges ] ──► [ Max Pooling ] ──► [ ResNet Blocks ] ──► Class Prediction
```

---

## 2. Core Concepts & Granular Step-by-Step Breakdown

### 1. The 2D Convolution Operation
A **Kernel / Filter** (e.g., $3 \times 3$ matrix) slides across an image matrix step-by-step (**Stride** $S$). At each step, it computes element-wise multiplication with the underlying image pixels and sums them up, outputting a single number into a new **Feature Map**.

- **Padding ($P$)**: Adding zero-borders around images so convolution doesn't shrink spatial dimensions.
- **Max Pooling**: Down-samples feature maps by extracting the maximum value in a window (e.g., $2 \times 2$), granting translation invariance (a cat shifted 2 pixels right is still recognized as a cat!).

---

### 2. The ResNet Revolution: Skip Connections

As researchers built deeper CNNs (50 to 100 layers), they hit a wall: **models got WORSE, not better!** 

Why? As gradients backpropagate backward through 50 layers of matrix multiplications, they shrink exponentially to zero (**Vanishing Gradient Problem**).

In 2015, He et al. introduced **ResNet (Residual Networks)** with a ridiculously simple trick: **The Skip Connection (Shortcut)**.

```
          x (Input Feature Map)
          │───────────────┐ (Identity Skip Connection / Shortcut)
          ▼               │
    [ 3x3 Conv2D ]        │
          │               │
      [ ReLU ]            │
          │               │
    [ 3x3 Conv2D ]        │
          │               │
          ▼               ▼
          [ Addition Node + ]  <-- Output = F(x) + x
                  │
              [ ReLU ]
```

Instead of forcing layers to learn a completely new target transformation $H(x)$, ResNet forces layers to learn only the small residual delta $F(x) = H(x) - x$, outputting $F(x) + x$. 

If a layer learns nothing ($F(x) = 0$), the identity shortcut passes input $x$ through untouched! Gradients flow directly backward along the identity addition highway without vanishing, allowing networks to scale to 152+ layers deep!

---

## 3. Practical Implementation: Building a ResNet Mini in PyTorch

Let's write a complete PyTorch script constructing a Residual CNN block and classifier:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Define Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip Connection Addition Highway: F(x) + x
        return F.relu(out)

# 2. Modern ConvNet Architecture
class ModernConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ModernConvNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.res_block1 = ResidualBlock(32)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.res_block1(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits

# Test model shape with synthetic image tensor (Batch=4, Channels=3, H=32, W=32)
model = ModernConvNet(num_classes=10)
dummy_image_batch = torch.randn(4, 3, 32, 32)
output_logits = model(dummy_image_batch)

print(f"Input Image Tensor Shape:  {dummy_image_batch.shape}")
print(f"Output Logits Tensor Shape: {output_logits.shape}")
```

---

## 4. Demystifying the Mathematics: Explaining Every Formula

Let's dissect the spatial output dimension formula for 2D convolutions.

### Equation 1: Spatial Output Dimension Formula
If you pass a $W \times W$ image into a convolution layer with kernel size $K$, padding $P$, and stride $S$, how big will the output feature map be?

$$
O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1
$$

- $W$: Input spatial width (e.g., $W = 224$ pixels).
- $K$: Filter kernel size (e.g., $K = 3$ for a $3 \times 3$ filter).
- $P$: Zero-padding width added around the border. Adding $P=1$ adds 1 pixel left and 1 pixel right, increasing width by $+2P = +2$.
- $W - K + 2P$: Total distance the kernel can slide across the padded image.
- $S$: Stride step size (e.g., $S = 2$ skips every other pixel).
- $\lfloor \dots \rfloor$: Floor function rounding down to the nearest integer.
- $+1$: Accounting for the initial starting placement of the filter on the first pixel.

---

## 5. Real-World Production Gotchas & Failure Modes

### Global Average Pooling vs. Fully Connected Layers
Classic networks (VGG-16) flattened 2D feature maps into massive 25,000-neuron dense fully-connected layers, creating 100+ million trainable parameters!
- *Fix*: Modern networks use `AdaptiveAvgPool2d((1, 1))` to compute the spatial mean average of each feature channel, reducing dense parameter count by $95\%$ while preventing severe overfitting.

---

## 6. Feynman Exercises & Deep Thinking Challenges

1. **Thought Experiment**: Calculate spatial output dimension for a $224 \times 224$ image passed through a $7 \times 7$ convolution with stride $S=2$ and padding $P=3$.
   - *Answer/Explanation*: $O = \lfloor \frac{224 - 7 + 2(3)}{2} \rfloor + 1 = \lfloor \frac{223}{2} \rfloor + 1 = 111 + 1 = 112 \times 112$.

2. **Code Challenge**: Build a PyTorch CNN and print tensor shapes at every step from input image to output logits.
