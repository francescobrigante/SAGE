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
