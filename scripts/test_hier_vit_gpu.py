"""End-to-end GPU smoke test for HierViT (cplx + real) on a single A100.

Loads the Hydra config `models/hiervit_cplx_x64`, instantiates the full
encoder+bottleneck+decoder via `hydra.utils.instantiate`, runs a forward
pass with a realistic batch (B=4, complex STFT 1025x128), backprops a
trivial loss, and checks shapes, dtypes, and gradient sanity.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def _section(title: str) -> None:
    print(f"\n========== {title} ==========")


def _check(name: str, cond: bool, *, fatal: bool = True) -> None:
    status = "OK  " if cond else "FAIL"
    print(f"[{status}] {name}")
    if fatal and not cond:
        sys.exit(1)


def make_cplx_models():
    from c_vae.hier_vit import HierViTEncoder, HierViTDecoder
    from c_vae.bottleneck import ComplexVAEBottleneck

    # Matches config/models/hiervit_cplx_x64.yaml (latent_channels=128 → ×16 compression).
    latent_channels = 128
    params_to_predict = 3
    enc = HierViTEncoder(
        in_channels=2, embed_dim=64, depths=[2, 2, 6, 2], num_heads=[4, 8, 16, 32],
        patch_size=(16, 4), dimension=latent_channels * params_to_predict, mlp_ratio=4.0,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
        is_complex=True, complex_activation="ComplexGELU1d",
        theta_y=1000.0, theta_x=10000.0,
    )
    dec = HierViTDecoder(
        channels=latent_channels, in_channels=2, embed_dim=64,
        depths=[2, 6, 2, 2], num_heads=[32, 16, 8, 4],
        patch_size=(16, 4), mlp_ratio=4.0,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
        is_complex=True, complex_activation="ComplexGELU1d",
        theta_y=1000.0, theta_x=10000.0,
    )
    btl = ComplexVAEBottleneck(proper=False)
    return enc, dec, btl


def make_real_models():
    from c_vae.hier_vit import HierViTEncoder, HierViTDecoder
    from ar_spectra.models.bottlenecks import VAEBottleneck

    enc = HierViTEncoder(
        in_channels=4, embed_dim=64, depths=[2, 2, 6, 2], num_heads=[4, 8, 16, 32],
        patch_size=(16, 4), dimension=32, mlp_ratio=4.0,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
        is_complex=False,
    )
    dec = HierViTDecoder(
        channels=16, in_channels=4, embed_dim=64, depths=[2, 6, 2, 2], num_heads=[32, 16, 8, 4],
        patch_size=(16, 4), mlp_ratio=4.0,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
        is_complex=False,
    )
    btl = VAEBottleneck(parameters_to_predict=2)
    return enc, dec, btl


def smoke_complex(device: torch.device) -> None:
    _section("COMPLEX path (cplx STFT, 2 channels)")
    enc, dec, btl = make_cplx_models()
    enc = enc.to(device).train()
    dec = dec.to(device).train()
    btl = btl.to(device).train()

    n_params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters())
    print(f"trainable params (enc+dec): {n_params/1e6:.2f} M")

    B = 4
    x = (torch.randn(B, 2, 1025, 128) + 1j * torch.randn(B, 2, 1025, 128)).to(torch.complex64).to(device)
    print(f"input  : {tuple(x.shape)}  {x.dtype}")

    z, info = enc(x)
    _check("encoder output is complex64", z.dtype == torch.complex64)
    _check("encoder output shape (B, 384, 32)", tuple(z.shape) == (B, 384, 32))
    _check("feature_shape (8, 4)", info["feature_shape"] == (8, 4))
    print(f"encoded: {tuple(z.shape)}  {z.dtype}")

    z_lat, bot_info = btl.encode(z, return_info=True)
    _check("bottleneck output is complex64", z_lat.dtype == torch.complex64)
    _check("bottleneck output shape (B, 128, 32)", tuple(z_lat.shape) == (B, 128, 32))
    _check("KL is finite", torch.isfinite(bot_info["kl"]).item())
    print(f"latent : {tuple(z_lat.shape)}  KL={bot_info['kl'].item():.4f}")

    y = dec(z_lat)
    _check("decoder output is complex64", y.dtype == torch.complex64)
    _check("decoder output shape (B, 2, 1024, 128)", tuple(y.shape) == (B, 2, 1024, 128))
    print(f"decoded: {tuple(y.shape)}  {y.dtype}")

    target = x[..., :1024, :]                                   # crop 1025→1024 to match decoder out
    recon = (y.real - target.real).pow(2).mean() + (y.imag - target.imag).pow(2).mean()
    loss = recon + 1e-3 * bot_info["kl"]
    loss.backward()
    print(f"loss   : recon={recon.item():.4f}  total={loss.item():.4f}")

    n_with_grad = sum(1 for p in enc.parameters() if p.grad is not None)
    n_grad_nan = sum(1 for p in enc.parameters() if p.grad is not None and torch.isnan(p.grad.abs()).any())
    _check(f"encoder params with grad ({n_with_grad}>0)", n_with_grad > 0)
    _check(f"encoder grads NaN-free ({n_grad_nan}=0)", n_grad_nan == 0)

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"GPU peak memory: {peak_mb:.0f} MiB")


def smoke_real(device: torch.device) -> None:
    _section("REAL path (CAC, 4 channels)")
    enc, dec, btl = make_real_models()
    enc = enc.to(device).train()
    dec = dec.to(device).train()
    btl = btl.to(device).train()

    n_params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters())
    print(f"trainable params (enc+dec): {n_params/1e6:.2f} M")

    B = 4
    x = torch.randn(B, 4, 1025, 128, device=device, requires_grad=True)
    print(f"input  : {tuple(x.shape)}  {x.dtype}")

    z, info = enc(x)
    _check("encoder output is float32", z.dtype == torch.float32)
    _check("encoder output shape (B, 32, 32)", tuple(z.shape) == (B, 32, 32))
    print(f"encoded: {tuple(z.shape)}")

    z_lat, bot_info = btl.encode(z, return_info=True)
    _check("real bottleneck output shape (B, 16, 32)", tuple(z_lat.shape) == (B, 16, 32))
    _check("KL is finite", torch.isfinite(bot_info["kl"]).item())
    print(f"latent : {tuple(z_lat.shape)}  KL={bot_info['kl'].item():.4f}")

    y = dec(z_lat)
    _check("decoder output shape (B, 4, 1024, 128)", tuple(y.shape) == (B, 4, 1024, 128))
    print(f"decoded: {tuple(y.shape)}")

    target = x[..., :1024, :]
    recon = (y - target).pow(2).mean()
    loss = recon + 1e-3 * bot_info["kl"]
    loss.backward()
    print(f"loss   : recon={recon.item():.4f}  total={loss.item():.4f}")

    n_with_grad = sum(1 for p in enc.parameters() if p.grad is not None)
    n_grad_nan = sum(1 for p in enc.parameters() if p.grad is not None and torch.isnan(p.grad.abs()).any())
    _check(f"encoder params with grad ({n_with_grad}>0)", n_with_grad > 0)
    _check(f"encoder grads NaN-free ({n_grad_nan}=0)", n_grad_nan == 0)


def smoke_hydra(device: torch.device) -> None:
    """Verify the Hydra config instantiates the model with the same shape contract."""
    _section("HYDRA config instantiate (hiervit_cplx_x64)")
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    cfg_path = REPO / "config" / "models" / "hiervit_cplx_x64.yaml"
    cfg = OmegaConf.load(cfg_path)
    # Resolve interpolations using a wrapper namespace so ${models.model...} refs work.
    full = OmegaConf.create({"models": cfg})
    OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b), replace=True)

    enc_cfg = OmegaConf.to_container(full.models.model.encoder, resolve=True)
    dec_cfg = OmegaConf.to_container(full.models.model.decoder, resolve=True)
    btl_cfg = OmegaConf.to_container(full.models.model.bottleneck, resolve=True)
    print("encoder _target_:", enc_cfg["_target_"])
    print("decoder _target_:", dec_cfg["_target_"])
    print("bottleneck _target_:", btl_cfg["_target_"])

    enc = instantiate(enc_cfg).to(device).train()
    dec = instantiate(dec_cfg).to(device).train()
    btl = instantiate(btl_cfg).to(device).train()

    B = 2
    x = (torch.randn(B, 2, 1025, 128) + 1j * torch.randn(B, 2, 1025, 128)).to(torch.complex64).to(device)
    z, info = enc(x)
    z_lat, bot_info = btl.encode(z, return_info=True)
    y = dec(z_lat)

    _check("hydra cplx encoder out (B, 384, 32)", tuple(z.shape) == (B, 384, 32))
    _check("hydra cplx bottleneck out (B, 128, 32)", tuple(z_lat.shape) == (B, 128, 32))
    _check("hydra cplx decoder out (B, 2, 1024, 128)", tuple(y.shape) == (B, 2, 1024, 128))
    print(f"shapes ok: enc {tuple(z.shape)}  bot {tuple(z_lat.shape)}  dec {tuple(y.shape)}")


def main() -> None:
    print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("WARN: no GPU detected — running on CPU")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        print("GPU:", torch.cuda.get_device_name(0))

    torch.manual_seed(0)
    smoke_complex(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    smoke_real(device)

    smoke_hydra(device)

    _section("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
