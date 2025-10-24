# 🎧 EulerAudioBackbone: Generative Backbone for Complex-Valued Audio Spectrograms

## Overview
**ComplexVAE** is a research-oriented backbone for **complex-valued Variational Autoencoders (VAE)** operating directly on **complex STFT spectrograms**.  
It is designed as part of a broader ICML project exploring **complex-valued neural architectures** for:
- Neural Audio Coding  
- Generative Modeling (VAE → Diffusion)  
- Source Separation  

Unlike prior models (e.g., [Stable Audio Open](https://huggingface.co/stabilityai/stable-audio-open-1.0), [Music2Latent2](https://anonymous2732.github.io/music2latent2/), [CoDiCodec](https://sonycslparis.github.io/codicodec)), which employ **real-valued** architectures on **complex inputs**, this repository investigates **end-to-end complex-valued** formulations as true backbones for generative models.

---

## 🔬 Research Goals
- Build and release **open-source complex-valued backbones** for audio representation learning.  
- Conduct systematic **ablation studies** on:
  - Real vs. Complex architectures (ConvNeXt, Conformer, etc.)
  - Reconstruction quality on complex spectrograms
  - Impact of complex-valued normalization, activations, and attention.
- Enable **generalizable backbones** across multiple audio domains (music, speech, ambient, etc.).

---

## 🧠 Architecture
- Encoder/Decoder designed for **complex STFT tensors** `(B, C, F, T)`  
- Supports both **complex-valued convolutions** and **attention-based variants**
- Modular structure for flexible integration into:
  - Latent diffusion models
  - Neural audio codecs
  - Source separation systems  

---

## ⚙️ Planned Experiments
| Task | Backbone | Domain | Evaluation |
|------|-----------|---------|-------------|
| Neural Audio Coding | EuleroDec / complex | Jamendo / DNS / LibriTTS | SI-SDR, FAD, ViSQOL |
| VAE for Diffusion | ComplexVAE | Multi-domain | Reconstruction, KL, FAD |
| Source Separation | Complex Swin  / BSRoformer-C | Music & Speech | SI-SDR, SDRi |

---

## 📚 Related Works
- **Stable Audio Open** — Latent diffusion on waveforms (44.1 kHz stereo).  
- **Music2Latent2** — Continuous autoencoder with summary embeddings and autoregressive decoding.  
- **CoDiCodec** — Unified continuous/discrete consistency model with FSQ and summary embeddings.  
This project extends these paradigms into the **complex domain**, enabling explicit phase modeling and analyticity constraints.

---

## 🧩 Repository Structure

# AutoEncoder Construction Guide

This project lets you build the AutoEncoder in multiple interchangeable ways. All approaches produce an AutoEncoder instance; pick the one that best fits your workflow or tooling.

## 1) From a dict “spec” (same shape as JSON configs)

```python
from ar_spectra.models.autoencoder import AutoEncoder

ae = AutoEncoder(
    encoder={
        "class": "ar_spectra.models.autoencoders.SeaNET_AE.SEANetEncoder2d",
        "kwargs": {"input_size": 128}
    },
    decoder={
        "class": "ar_spectra.models.autoencoders.SeaNET_AE.SEANetDecoder2d",
        "kwargs": {"input_size": 64}
    },
    bottleneck={
        "class": "ar_spectra.models.bottlenecks.VAEBottleneck",
        "kwargs": {"skip_bottleneck": True}
    }
)
```

## 2) Passing classes directly (no constructor args)

```python
from ar_spectra.models.autoencoders.SeaNET_AE import SEANetEncoder2d, SEANetDecoder2d
from ar_spectra.models.autoencoder import AutoEncoder

ae = AutoEncoder(encoder=SEANetEncoder2d, decoder=SEANetDecoder2d)
```

## 3) Passing class paths as strings (no constructor args)

```python
from ar_spectra.models.autoencoder import AutoEncoder

ae = AutoEncoder(
    encoder="ar_spectra.models.autoencoders.SeaNET_AE.SEANetEncoder2d",
    decoder="ar_spectra.models.autoencoders.SeaNET_AE.SEANetDecoder2d",
)
```

## 4) Passing prebuilt instances

```python
from ar_spectra.models.autoencoders.SeaNET_AE import SEANetEncoder2d, SEANetDecoder2d
from ar_spectra.models.autoencoder import AutoEncoder

enc = SEANetEncoder2d(input_size=128)
dec = SEANetDecoder2d(input_size=64)
ae = AutoEncoder(encoder=enc, decoder=dec)
```

## 5) From a model config dict (recommended with experiment JSON)

```python
from ar_spectra.models.autoencoder import AutoEncoder

# cfg is your experiment config loaded from JSON (e.g., ar_spectra/config/experiments/SEANet_STFT.json)
ae = AutoEncoder.from_config(cfg["model"])
```

---

## How the pieces work together

- AutoEncoder.encode(x): runs encoder(x) → optional bottleneck.encode(latents) → latents.
- AutoEncoder.decode(z): runs optional bottleneck.decode(z) → decoder(z) → reconstruction.
- AutoEncoder.istft(spec, ...): converts complex/RI spectrogram back to waveform using provided STFT params.

### Dimension checks
To prevent silent shape mismatches, the constructor validates channels:
- Without a VAE bottleneck: encoder output channels must equal decoder input channels.
- With VAEBottleneck: encoder channels must be 2 × decoder input (mean + logvar).
- With SkipBottleneck: encoder channels must equal decoder input.

---
