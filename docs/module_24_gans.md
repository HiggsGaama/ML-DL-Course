# Module 24: Generative Adversarial Networks (GANs)

## 1. The Physical Intuition: The Art Forger and the Detective Game

Imagine a high-stakes game between two people:
1. **The Forger (Generator $G$)**: An art counterfeiter who has never seen a real Picasso painting in their life. They try to paint fake artwork from random noise sketches and sell them to a museum.
2. **The Detective (Discriminator $D$)**: An art expert who inspects paintings and tries to determine whether a painting is a genuine Picasso or a cheap forgery.

```
Noise Latent z ──► [ Generator G ] ──► Fake Image x_fake ──┐
                                                           ├──► [ Discriminator D ] ──► Probability Real/Fake
Real Images ─────────────────────────► Real Image x_real ──┘
```

At the beginning of training, both are terrible. The Forger paints random colorful splotches. The Detective flips a coin.

As training progresses:
- The Detective inspects real Picassos, catches the Forger's crude splotches, and rejects them.
- The Forger receives feedback (*"The colors were wrong"*), adjusts their painting techniques, and creates better fakes.
- The Detective is forced to look closer at micro-brushstrokes to catch the new fakes.

This adversarial game is a **Minimax Zero-Sum Game**. Eventually, the Forger becomes so masterfully skilled that their generated images are physically indistinguishable from real paintings, and the Detective can only guess with 50% probability!

---

## 2. Core Concepts & Granular Step-by-Step Breakdown

### 1. The Minimax Objective Equation
The Generator $G$ and Discriminator $D$ compete to optimize a single shared value function $V(D, G)$:

$$\min_{G} \max_{D} V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}} [\log D(x)] + \mathbb{E}_{z \sim p_{z}} [\log (1 - D(G(z)))]$$

- $\max_D$: The Discriminator wants to maximize $D(x)$ to $1.0$ for real images and minimize $D(G(z))$ to $0.0$ for fake images.
- $\min_G$: The Generator wants to minimize this value by fooling the Discriminator into predicting $D(G(z)) = 1.0$!

---

### 2. Major GAN Training Failure Modes

1. **Mode Collapse**: The Generator discovers one single fake image that tricks the Discriminator (e.g., a specific cat face) and outputs that exact same image repeatedly for every random noise vector $z$, losing all variety.
2. **Vanishing Gradients**: If the Discriminator gets too smart too fast, $D(G(z)) \to 0$ constantly, causing Generator gradients to vanish so it stops learning.
3. **Wasserstein GAN (WGAN)**: Replaces log-loss with Earth Mover's Distance (Wasserstein Distance), creating smooth non-saturating gradients everywhere!

---

## 3. Architecture & Visual Diagrams

### DCGAN (Deep Convolutional GAN) Topology

```
Latent Noise Vector z (100) ──► [ ConvTranspose2D Up-sampling ] ──► Generator Output Image (64x64x3)
                                                                            │
                                                                            ▼
Real / Generated Images     ──► [ Conv2D Down-sampling ]        ──► Discriminator Probability (0.0 to 1.0)
```

---

## 4. Practical Implementation: Building a DCGAN in PyTorch

Let's write a complete PyTorch script building a Deep Convolutional Generator and Discriminator:

```python
import torch
import torch.nn as nn

# 1. Generator Network: Transposed Convolutions upsampling noise (100) -> Image (1x28x28)
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 7x7 -> 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),   # 14x14 -> 28x28
            nn.Tanh() # Normalizes output pixels to [-1, 1]
        )
        
    def forward(self, z):
        return self.net(z)

# 2. Discriminator Network: Standard Convolutions downsampling Image -> Probability
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1), # 28x28 -> 14x14
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 14x14 -> 7x7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid() # Probability output [0.0, 1.0]
        )
        
    def forward(self, img):
        return self.net(img)

# Test Execution
netG = Generator(latent_dim=100)
netD = Discriminator()

z = torch.randn(4, 100) # Mini-batch of 4 random noise vectors
fake_images = netG(z)
d_probs = netD(fake_images)

print(f"Noise Vector Shape:         {z.shape}")
print(f"Generated Fake Image Shape: {fake_images.shape}")
print(f"Discriminator Output Probs: {d_probs.squeeze().detach().numpy().round(4)}")
```

---

## 5. Demystifying the Mathematics: Explaining Every Formula

Let's dissect the minimax objective terms.

### Equation 1: GAN Value Function
$$
V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}} [\log D(x)] + \mathbb{E}_{z \sim p_{z}} [\log (1 - D(G(z)))]
$$

- $\mathbb{E}_{x \sim p_{\text{data}}} [\log D(x)]$: Expected log probability assigned by Discriminator to true training set images $x$.
- $\mathbb{E}_{z \sim p_{z}} [\log (1 - D(G(z)))]$: Expected log probability that the Discriminator correctly identifies generated image $G(z)$ as fake.
- **The Equilibrium Goal**: At optimal convergence, $D(x) = 0.5$ for all generated images, meaning the Discriminator can no longer distinguish real from fake!

---

## 6. Real-World Production Gotchas & Failure Modes

### Non-Saturating Generator Loss
In early training, $D(G(z)) \approx 0$, causing $\log(1 - D(G(z)))$ gradients to saturate and vanish.
- *Fix*: Instead of minimizing $\log(1 - D(G(z)))$, train the Generator to **maximize $\log D(G(z))$**, providing strong, non-saturating gradients early on!

---

## 7. Feynman Exercises & Deep Thinking Challenges

1. **Thought Experiment**: Why do we use `Tanh()` as the final activation layer in Generators instead of `Sigmoid()`?
   - *Answer/Explanation*: `Tanh()` bounds pixel outputs zero-centered between $[-1, 1]$, matching standard zero-centered normalized training data distributions and providing stronger symmetric gradients during training.

2. **Exercise**: Train the DCGAN script on MNIST digits for 5 epochs and display generated digit images.
