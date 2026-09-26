"""
Test: can the MRSTFT gradient flow through iSTFT to correct a global 
scale factor applied to the spectrogram (as the decoder would output)?

This tests the exact sage pipeline:
  spec * alpha → inverse_pre_transform → istft → MRSTFT(decoded_wav, gt_wav)

If ∂L/∂alpha is zero or negligible, the gradient is being blocked somewhere
in the iSTFT chain.
"""
import torch
import sys

from sage.nn.losses.spectral import MultiResolutionSpectrogramLoss
from sage.model.autoencoder import SAGEAutoencoder
import torch.nn as nn


def test_gradient_through_istft():
    """Test gradient of MRSTFT flowing through iSTFT to a scale factor on spec."""
    print("=" * 70)
    print("Test: MRSTFT gradient through iSTFT → spectrogram scale factor")
    print("=" * 70)

    # Setup autoencoder just for STFT/iSTFT
    class DummyEncoder(nn.Module):
        def forward(self, x): return x, {}
    class DummyDecoder(nn.Module):
        def forward(self, x, encoder_info=None): return x

    ae = SAGEAutoencoder(encoder=DummyEncoder(), decoder=DummyDecoder())
    ae.set_stft_config({"n_fft": 2048, "hop_length": 512, "win_length": 2048})

    mrstft = MultiResolutionSpectrogramLoss(
        fft_sizes=[2048, 1024, 512, 256, 128, 64],
        hop_sizes=[512, 256, 128, 64, 32, 16],
        factor_sc=1.0, factor_mag=1.0,
        log_mag=True, factor_log_mag=1.0,
    )

    torch.manual_seed(42)
    B, C, T = 2, 2, 65024
    wav_gt = torch.randn(B, C, T) * 0.5

    # Get ground truth spectrogram
    spec_gt = ae.stft(wav_gt)  # (B, C, 1025, 128) complex
    
    # Simulate Swin crop: 1025 → 1024
    spec_gt_cropped = spec_gt[..., :1024, :].detach()

    print(f"\n{'α':>10s} | {'Loss':>12s} | {'∂L/∂α (spec)':>18s} | {'α·∂L/∂α':>15s}")
    print("-" * 65)

    for alpha_val in [0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.5]:
        alpha = torch.tensor(alpha_val, requires_grad=True)
        
        # Scale the spectrogram (simulating decoder output)
        spec_scaled = spec_gt_cropped * alpha  # complex * real
        
        # iSTFT (through _maybe_add_nyquist)
        wav_recon = ae.istft(spec_scaled, target_length=T)
        
        # Trim
        if wav_recon.shape[-1] > wav_gt.shape[-1]:
            wav_recon = wav_recon[..., :T]
        elif wav_recon.shape[-1] < wav_gt.shape[-1]:
            wav_gt_trim = wav_gt[..., :wav_recon.shape[-1]]
        else:
            wav_gt_trim = wav_gt
        
        # MRSTFT loss
        loss = mrstft(wav_recon, wav_gt_trim.detach())
        loss.backward()
        
        grad = alpha.grad.item()
        print(f"{alpha_val:10.2f} | {loss.item():12.6f} | {grad:18.8f} | {alpha_val * grad:15.8f}")

    print()
    print("If ∂L/∂α is strong at α=0.8, the gradient flows correctly through iSTFT.")
    print("If ∂L/∂α is ~0 at α=0.8, the iSTFT blocks the gradient.")


def test_gradient_through_istft_with_pretransform():
    """Test with PowerMagnitudeTransform in the chain (as in real training)."""
    print()
    print("=" * 70)
    print("Test: MRSTFT gradient through pre_transform + iSTFT")
    print("=" * 70)

    from sage.nn.pre_transform import PowerMagnitudeTransform
    
    class DummyEncoder(nn.Module):
        def forward(self, x): return x, {}
    class DummyDecoder(nn.Module):
        def forward(self, x, encoder_info=None): return x

    ae = SAGEAutoencoder(encoder=DummyEncoder(), decoder=DummyDecoder(),
                     pre_transform={"type": "power_norm", "config": {"alpha": 0.65, "beta": 0.35}})
    ae.set_stft_config({"n_fft": 2048, "hop_length": 512, "win_length": 2048})

    mrstft = MultiResolutionSpectrogramLoss(
        fft_sizes=[2048, 1024, 512, 256, 128, 64],
        hop_sizes=[512, 256, 128, 64, 32, 16],
        factor_sc=1.0, factor_mag=1.0,
        log_mag=True, factor_log_mag=1.0,
    )

    torch.manual_seed(42)
    B, C, T = 2, 2, 65024
    wav_gt = torch.randn(B, C, T) * 0.5
    
    spec_gt = ae.stft(wav_gt)
    spec_gt_cropped = spec_gt[..., :1024, :].detach()
    
    # Apply pre_transform (as encoder would)
    spec_transformed = ae._apply_pre_transform(spec_gt_cropped).detach()

    print(f"\n{'α':>10s} | {'Loss':>12s} | {'∂L/∂α':>18s}")
    print("-" * 48)

    for alpha_val in [0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.5]:
        alpha = torch.tensor(alpha_val, requires_grad=True)
        
        # Scale in the transformed domain (simulating decoder output)
        spec_scaled = spec_transformed * alpha
        
        # Inverse pre_transform
        spec_linear = ae._apply_inverse_pre_transform(spec_scaled)
        
        # iSTFT
        wav_recon = ae.istft(spec_linear, target_length=T)
        
        wav_gt_trim = wav_gt[..., :wav_recon.shape[-1]]
        
        loss = mrstft(wav_recon, wav_gt_trim.detach())
        loss.backward()
        
        grad = alpha.grad.item()
        print(f"{alpha_val:10.2f} | {loss.item():12.6f} | {grad:18.8f}")

    print()
    print("Same test but with PowerMag inverse in the gradient chain.")


def test_gradient_through_pack_unpack_istft():
    """Test with CAC (complex-as-channels) pack/unpack in the chain."""
    print()
    print("=" * 70)
    print("Test: MRSTFT gradient through CAC pack/unpack + iSTFT")
    print("=" * 70)

    class DummyEncoder(nn.Module):
        def forward(self, x): return x, {}
    class DummyDecoder(nn.Module):
        def forward(self, x, encoder_info=None): return x

    ae = SAGEAutoencoder(encoder=DummyEncoder(), decoder=DummyDecoder())
    ae.set_stft_config({"n_fft": 2048, "hop_length": 512, "win_length": 2048})

    mrstft = MultiResolutionSpectrogramLoss(
        fft_sizes=[2048, 1024, 512, 256, 128, 64],
        hop_sizes=[512, 256, 128, 64, 32, 16],
        factor_sc=1.0, factor_mag=1.0,
        log_mag=True, factor_log_mag=1.0,
    )

    torch.manual_seed(42)
    B, C, T = 2, 2, 65024
    wav_gt = torch.randn(B, C, T) * 0.5
    
    spec_gt = ae.stft(wav_gt)  # complex
    spec_gt_cropped = spec_gt[..., :1024, :]
    
    # Pack complex → real CAC (as dataloader does with cac=True)
    packed = ae._pack_complex(spec_gt_cropped).detach()  # (B, 4, 1024, 128) float32

    print(f"\n{'α':>10s} | {'Loss':>12s} | {'∂L/∂α':>18s}")
    print("-" * 48)

    for alpha_val in [0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.5]:
        alpha = torch.tensor(alpha_val, requires_grad=True)
        
        # Scale in packed domain (what the model decoder outputs)
        packed_scaled = packed * alpha
        
        # Unpack to complex
        spec_complex = ae._unpack_complex(packed_scaled)
        
        # iSTFT
        wav_recon = ae.istft(spec_complex, target_length=T)
        
        wav_gt_trim = wav_gt[..., :wav_recon.shape[-1]]
        
        loss = mrstft(wav_recon, wav_gt_trim.detach())
        loss.backward()
        
        grad = alpha.grad.item()
        print(f"{alpha_val:10.2f} | {loss.item():12.6f} | {grad:18.8f}")

    print()
    print("If gradients are strong here too, the pack/unpack is not blocking.")


def test_direct_waveform_vs_istft_gradient_strength():
    """Compare gradient magnitude: direct waveform loss vs iSTFT chain."""
    print()
    print("=" * 70)
    print("COMPARISON: Direct waveform α vs iSTFT-chain α gradient strength")
    print("=" * 70)

    class DummyEncoder(nn.Module):
        def forward(self, x): return x, {}
    class DummyDecoder(nn.Module):
        def forward(self, x, encoder_info=None): return x

    ae = SAGEAutoencoder(encoder=DummyEncoder(), decoder=DummyDecoder())
    ae.set_stft_config({"n_fft": 2048, "hop_length": 512, "win_length": 2048})

    mrstft = MultiResolutionSpectrogramLoss(
        fft_sizes=[2048, 1024, 512, 256, 128, 64],
        hop_sizes=[512, 256, 128, 64, 32, 16],
        factor_sc=1.0, factor_mag=1.0,
        log_mag=True, factor_log_mag=1.0,
    )

    torch.manual_seed(42)
    B, C, T = 2, 2, 65024
    wav_gt = torch.randn(B, C, T) * 0.5
    spec_gt = ae.stft(wav_gt)
    spec_cropped = spec_gt[..., :1024, :].detach()

    alpha_val = 0.8

    # Test 1: Direct waveform scale (SAO-like)
    alpha_direct = torch.tensor(alpha_val, requires_grad=True)
    wav_scaled = wav_gt.detach() * alpha_direct
    loss_direct = mrstft(wav_scaled, wav_gt.detach())
    loss_direct.backward()
    grad_direct = alpha_direct.grad.item()

    # Test 2: Spectrogram scale → iSTFT (sage-like)
    alpha_istft = torch.tensor(alpha_val, requires_grad=True)
    spec_scaled = spec_cropped * alpha_istft
    wav_recon = ae.istft(spec_scaled, target_length=T)
    wav_gt_trim = wav_gt[..., :wav_recon.shape[-1]]
    loss_istft = mrstft(wav_recon, wav_gt_trim.detach())
    loss_istft.backward()
    grad_istft = alpha_istft.grad.item()

    print(f"\n  At α = {alpha_val}:")
    print(f"  SAO-like (direct wav scale):     loss={loss_direct.item():.6f}, ∂L/∂α = {grad_direct:.8f}")
    print(f"  sage (spec→iSTFT scale):   loss={loss_istft.item():.6f}, ∂L/∂α = {grad_istft:.8f}")
    print(f"  Gradient ratio (SAGE/SAO): {grad_istft/grad_direct:.4f}")
    print()
    if abs(grad_istft / grad_direct) > 0.9:
        print("  ✓ Gradients are comparable — iSTFT is NOT blocking the scale signal.")
    elif abs(grad_istft / grad_direct) > 0.1:
        print("  ⚠ Gradients are weaker through iSTFT but still significant.")
    else:
        print("  ✗ Gradient is severely attenuated through iSTFT!")


if __name__ == "__main__":
    test_gradient_through_istft()
    test_gradient_through_istft_with_pretransform()
    test_gradient_through_pack_unpack_istft()
    test_direct_waveform_vs_istft_gradient_strength()
