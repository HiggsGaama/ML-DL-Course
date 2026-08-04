# Module 26: Latent Diffusion Models & Stable Diffusion

## 1. The Physical Intuition: Denoising in Compressed Latent Space

Imagine trying to run a 1,000-step diffusion model directly on a high-resolution $1024 \times 1024 \times 3$ RGB photograph.

Every image contains **3,145,728 numbers**! Running a 100-layer U-Net across 3,000,000 floating-point numbers for 50 steps requires a $\$50,000$ supercomputer cluster. It is completely unviable for consumer hardware.

In 2022, Rombach et al. introduced **Latent Diffusion Models (LDM / Stable Diffusion)** with a brilliant realization:

Most pixels in a high-res photo are redundant noise (e.g., thousands of identical blue pixels in a clear sky). What if you use a Variational Autoencoder (VAE) to compress the $512 \times 512 \times 3$ image by **$8\times$ spatially**, shrinking it into a compact $64 \times 64 \times 4$ **Latent Space** representation?

```
Pixel Space (512x512x3 = 786,432 numbers) ──► [ VAE Encoder ] ──► Latent Space z (64x64x4 = 16,384 numbers)
                                                                            │
                                                                            ▼
Image Output (512x512x3) ◄── [ VAE Decoder ] ◄── Denoised Latent z ◄── [ Latent Diffusion U-Net ]
```

By running the diffusion denoising process **strictly inside compressed latent space**, computational compute requirements drop by **$48\times$**, allowing high-res image generation on a single consumer GPU in seconds!

---

## 2. Core Concepts & Granular Step-by-Step Breakdown

### 1. The Three Core Pillars of Stable Diffusion

1. **Variational Autoencoder (VAE)**: Compresses images into latent space $z = E(x)$, and decodes denoised latents back into high-res pixels $\hat{x} = D(z)$.
2. **CLIP Text Encoder**: Converts human text prompts (e.g., `"A cyberpunk city in rain"`) into text conditioning vectors $y = \text{CLIP}(c)$.
3. **Conditioned U-Net Denoiser**: Uses **Cross-Attention layers** to inject the text conditioning vectors $y$ into the denoising process!

---

### 2. Classifier-Free Guidance (CFG) Scale
How do you force the U-Net to follow your text prompt strictly instead of ignoring it?

During sampling, we calculate TWO noise predictions:
- $\epsilon_{\text{uncond}} = \epsilon_\theta(z_t, t, \emptyset)$ (Unconditioned noise prediction).
- $\epsilon_{\text{cond}} = \epsilon_\theta(z_t, t, c)$ (Text-conditioned noise prediction).

We combine them using the **CFG Scale $w$**:

$$\hat{\epsilon} = \epsilon_{\text{uncond}} + w \cdot \left( \epsilon_{\text{cond}} - \epsilon_{\text{uncond}} \right)$$

- If $w = 1.0$: Standard text-conditioned sampling.
- If $w = 7.5$ (Standard Default): Amplifies the difference vector pointing toward text alignment, forcing the model to follow the prompt aggressively!

---

## 3. Practical Implementation: Local Generation with Diffusers

Let's write a clean Python script demonstrating Stable Diffusion generation using Hugging Face `diffusers`:

```python
import torch
from diffusers import StableDiffusionPipeline

def generate_image_from_prompt(prompt: str, output_path: str = "output.png"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing Stable Diffusion generation on device: {device}")
    
    # 1. Load Pre-trained Stable Diffusion Pipeline (v1-5 / v2-1)
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)
    
    # 2. Configure Seed for Reproducibility
    generator = torch.Generator(device=device).manual_seed(42)
    
    # 3. Execute Latent Denoising Pipeline
    image = pipe(
        prompt=prompt,
        negative_prompt="blurry, low quality, distorted",
        num_inference_steps=25,   # Fast sampling using DDIM/DPMSolver
        guidance_scale=7.5,      # Classifier-Free Guidance Scale
        generator=generator
    ).images[0]
    
    image.save(output_path)
    print(f"Successfully generated and saved image to '{output_path}'!")

# Demonstration Run (Uncomment in GPU environment)
# generate_image_from_prompt("A cinematic shot of a robotic engineer in a neon laboratory")
```

---

## 4. Demystifying the Mathematics: Explaining Every Formula

Let's dissect the Classifier-Free Guidance (CFG) formula.

### Equation 1: Classifier-Free Guidance Update Formula
$$
\hat{\epsilon}_\theta(z_t, c) = \epsilon_\theta(z_t, \emptyset) + w \cdot \Big( \epsilon_\theta(z_t, c) - \epsilon_\theta(z_t, \emptyset) \Big)
$$

- $\epsilon_\theta(z_t, \emptyset)$: Baseline unconditioned noise prediction (what the model generates naturally without text).
- $\epsilon_\theta(z_t, c)$: Conditioned noise prediction (what the model generates with text prompt $c$).
- $\epsilon_\theta(z_t, c) - \epsilon_\theta(z_t, \emptyset)$: The **Directional Difference Vector** pointing towards the text prompt.
- $w$: CFG scale weight. Scaling $w > 1.0$ pushes the generation vector aggressively along the text directional vector, sharpening details matching your prompt!

---

## 5. Real-World Production Gotchas & Failure Modes

### Memory Management in Production (VRAM Allocation)
Loading VAE, U-Net, and CLIP models simultaneously requires 8GB+ VRAM.
- *Fixes*: Enable `pipe.enable_sequential_cpu_offload()` and `pipe.enable_attention_slicing()` to run Stable Diffusion on 4GB VRAM GPUs!

---

## 6. Feynman Exercises & Deep Thinking Challenges

1. **Thought Experiment**: What happens to image quality and prompt fidelity if you increase CFG scale $w$ to an extreme value like $w = 30.0$?
   - *Answer/Explanation*: Extreme CFG values force latents past learned distribution bounds, causing oversaturated colors, high-frequency artifacts, and burned image appearances.

2. **Exercise**: Modify the script to pass a custom negative prompt and observe how it suppresses unwanted visual artifacts.
