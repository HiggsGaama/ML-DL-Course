# Module 25: Diffusion Models Foundations (DDPM)

## 1. The Physical Intuition: Un-scrambling the Egg

Imagine taking a crystal-clear high-definition photograph of a Mona Lisa painting and dropping a single tiny drop of black ink into the center of the image. 

Now, add another drop of ink, and another, and another. Over 1,000 steps, ink diffuses across the canvas until the painting is destroyed, leaving behind pure, uniform random static noise. This is the **Forward Noising Process ($q$)**.

Now, ask yourself a wild question: **Can you build a neural network that watches this noise and learns how to remove a single drop of ink at a time?**

```
Clean Image x_0 ──► Add Gaussian Noise ──► x_1 ──► ... ──► Pure Noise x_T (Forward Noising Process)
                                                                 │
Clean Output x_0 ◄── U-Net Predicts Noise ◄── x_t-1 ◄── ... ◄───┘ (Reverse Denoising Process)
```

If a neural network (typically a U-Net) can learn to remove a single micro-step of noise, you can hand it a canvas of **pure random Gaussian noise generated from thin air ($x_T \sim \mathcal{N}(0, I)$)**. Over 1,000 reverse steps, the U-Net gradually scrubs away the noise, revealing a brand-new, ultra-photorealistic image created from nothing! 

This is **Denoising Diffusion Probabilistic Modeling (DDPM)**—the core foundation powering Stable Diffusion, Midjourney, and DALL-E 3!

---

## 2. Core Concepts & Granular Step-by-Step Breakdown

### 1. The Forward Process ($q$): Closed-Form Noise Injection
In the forward process, we add tiny amounts of Gaussian noise at each timestep $t \in [1, T]$ according to a noise variance schedule $\beta_t$:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I\right)$$

- **The Closed-Form Jump Sampling Trick**: Do we have to run all 1,000 steps sequentially to create a noisy image at step $t$? NO! 
- Using algebra, we can jump directly from clean image $x_0$ to any noisy step $x_t$ in a single equation:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon \quad \text{where } \epsilon \sim \mathcal{N}(0, I)$$

Where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$.

---

### 2. The Reverse Process ($p_\theta$): Learning to Denoise
The network (U-Net) doesn't try to predict the clean image $x_0$ directly. Instead, it predicts the **exact noise vector $\epsilon_\theta(x_t, t)$** that was added to image $x_0$ at timestep $t$!

The training objective simplifies to standard **Mean Squared Error (MSE)** between true added noise $\epsilon$ and predicted noise $\epsilon_\theta$:

$$\mathcal{L}_{\text{Simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \left\| \epsilon - \epsilon_\theta\left(x_t, t\right) \right\|^2 \right]$$

---

## 3. Architecture & Visual Diagrams

### DDPM Forward Jump Sampling and Reverse U-Net Denoising

```
FORWARDFULL JUMP:
Clean Image x_0 ──────────────────────────────────────────► Noisy Image x_t (via closed-form formula)
                                                                 │
REVERSE DENOISING STEP:                                          │
                                                                 ▼
[ U-Net Noise Predictor ε_θ(x_t, t) ] ◄──────────────────────────┘
             │
             ▼
Predicted Noise Tensor ε_pred ──► Subtract from x_t ──► Denoised Image x_t-1
```

---

## 4. Practical Implementation: DDPM Noise Schedule & Loss

Let's write a complete PyTorch script demonstrating the closed-form forward noise process and DDPM loss calculation:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Define Linear Noise Schedule (T=1000 steps)
T = 1000
beta = torch.linspace(1e-4, 0.02, T) # Linear variance schedule
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0) # Accumulated product alpha_bar_t

# 2. Closed-Form Forward Jump Function: Samples x_t given x_0, timestep t, and noise epsilon
def q_sample(x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
        
    # Extract alpha_bar_t for batch of timesteps t
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar[t]).view(-1, 1, 1, 1)
    
    # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise, noise

# 3. Simulate Forward Pass Training Batch
x_0 = torch.randn(4, 3, 64, 64) # Batch of 4 clean images
t = torch.randint(0, T, (4,))   # Random timesteps for each sample in batch

noisy_x_t, true_noise = q_sample(x_0, t)

print("=== DDPM FORWARD NOISING SIMULATION ===")
print("Clean Image Batch Shape: ", x_0.shape)
print("Timesteps Sampled:       ", t.tolist())
print("Noisy Image x_t Shape:   ", noisy_x_t.shape)
print("True Added Noise Shape:  ", true_noise.shape)
```

---

## 5. Demystifying the Mathematics: Explaining Every Formula

Let's dissect the closed-form jump formula parameters.

### Equation 1: Closed-Form Forward Sampling
$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon
$$

- $x_0$: Original clean image tensor.
- $\epsilon \sim \mathcal{N}(0, I)$: Standard Gaussian noise tensor.
- $\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$: Accumulated signal retention. 
  - At $t=0$, $\bar{\alpha}_0 \approx 1.0 \to \sqrt{1.0} x_0 + \sqrt{0} \epsilon = x_0$ (100% clean image).
  - At $t=1000$, $\bar{\alpha}_{1000} \approx 0.0 \to \sqrt{0} x_0 + \sqrt{1.0} \epsilon = \epsilon$ (100% pure random noise!).

---

## 6. Real-World Production Gotchas & Failure Modes

### Sampling Speed Bottleneck (DDPM vs. DDIM)
Standard DDPM sampling requires running the U-Net 1,000 sequential times to generate a single image, taking 30+ seconds!
- *Fix*: Use non-Markovian samplers like **DDIM (Denoising Diffusion Implicit Models)** or **DPMSolver**, which reduce reverse sampling steps from 1,000 down to **20–50 steps** with zero quality loss!

---

## 7. Feynman Exercises & Deep Thinking Challenges

1. **Thought Experiment**: Why does the U-Net take timestep $t$ as an explicit input argument (`unet(x_t, t)`)?
   - *Answer/Explanation*: The U-Net must adjust its noise estimation scale depending on where it sits in the diffusion process; at $t=999$ the image is pure noise, whereas at $t=10$ the image is nearly clean. Timestep $t$ is passed via Sinusoidal Positional Embeddings.

2. **Exercise**: Plot $\sqrt{\bar{\alpha}_t}$ vs $\sqrt{1 - \bar{\alpha}_t}$ over $t \in [0, 1000]$ in Python using `matplotlib` to visualize how signal fades as noise increases.
