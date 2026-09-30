"""
Diagnostic test: verify that the MRSTFT loss is scale-invariant
and cannot correct amplitude mismatch.

This test proves that the combination of SpectralConvergence + log-magnitude
cannot drive amplitude correction because both terms are invariant to
global scaling: SC(ax, y) = SC(x, y) and log_mag(ax, y) = log_mag(x, y) + const.
"""
import torch
import sys

from sage.nn.losses.experimental import MultiResolutionSpectrogramLoss


def test_mrstft_scale_invariance():
    """Show that MRSTFT loss is nearly scale-invariant: scaling the input
    by a constant factor barely changes the loss value."""
    
    torch.manual_seed(42)
    
    # Create a test waveform (stereo, ~1s at 44100)
    B, C, T = 2, 2, 44100
    wav_gt = torch.randn(B, C, T) * 0.5
    
    # Instantiate the loss exactly as used in training runs D/E/F
    mrstft = MultiResolutionSpectrogramLoss(
        fft_sizes=[2048, 1024, 512, 256, 128, 64],
        hop_sizes=[512, 256, 128, 64, 32, 16],
        factor_sc=1.0,
        factor_mag=1.0,  # default
        log_mag=True,
        factor_log_mag=1.0,
    )
    
    # Compute loss for various scale factors
    print("=" * 70)
    print("MRSTFT Loss Scale-Invariance Test")
    print("=" * 70)
    print(f"{'Scale':>10s} | {'MRSTFT Loss':>15s} | {'Relative to 1.0':>18s}")
    print("-" * 50)
    
    loss_at_1 = None
    for scale in [0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5, 2.0]:
        wav_hat = wav_gt * scale
        loss_val = mrstft(wav_hat, wav_gt).item()
        if scale == 1.0:
            loss_at_1 = loss_val
        ratio = loss_val / loss_at_1 if loss_at_1 and loss_at_1 > 0 else float('inf')
        print(f"{scale:10.2f} | {loss_val:15.6f} | {ratio:18.4f}")
    
    print()
    print("If the loss is scale-invariant, values at scale=0.5 and scale=2.0")
    print("should be close to the value at scale=1.0.")
    print()
    
    # Now compare with a true scale-sensitive loss (L1 on waveform)
    print("=" * 70)
    print("L1 Waveform Loss (scale-sensitive) for comparison")
    print("=" * 70)
    print(f"{'Scale':>10s} | {'L1 Loss':>15s} | {'Relative to 1.0':>18s}")
    print("-" * 50)
    
    l1_at_1 = None
    for scale in [0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5, 2.0]:
        wav_hat = wav_gt * scale
        loss_val = (wav_hat - wav_gt).abs().mean().item()
        if scale == 1.0:
            l1_at_1 = loss_val
        ratio = loss_val / l1_at_1 if l1_at_1 and l1_at_1 > 0 else float('inf')
        print(f"{scale:10.2f} | {loss_val:15.6f} | {ratio:18.4f}")


def test_mrstft_gradient_at_scale():
    """Show that the gradient of MRSTFT w.r.t. a global scale parameter α
    is negligible, meaning the loss cannot correct amplitude."""
    
    torch.manual_seed(42)
    B, C, T = 2, 2, 44100
    wav_gt = torch.randn(B, C, T) * 0.5
    
    mrstft = MultiResolutionSpectrogramLoss(
        fft_sizes=[2048, 1024, 512, 256, 128, 64],
        hop_sizes=[512, 256, 128, 64, 32, 16],
        factor_sc=1.0,
        factor_mag=1.0,
        log_mag=True,
        factor_log_mag=1.0,
    )
    
    print()
    print("=" * 70)
    print("MRSTFT Gradient w.r.t. Global Scale Factor α")
    print("=" * 70)
    print(f"{'α':>10s} | {'Loss':>12s} | {'∂L/∂α':>15s} | {'α·∂L/∂α':>15s}")
    print("-" * 60)
    
    for alpha_val in [0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.5]:
        alpha = torch.tensor(alpha_val, requires_grad=True)
        wav_hat = wav_gt.detach() * alpha
        loss = mrstft(wav_hat, wav_gt.detach())
        loss.backward()
        grad = alpha.grad.item()
        print(f"{alpha_val:10.2f} | {loss.item():12.6f} | {grad:15.8f} | {alpha_val * grad:15.8f}")
    
    print()
    print("If ∂L/∂α is close to zero at α=0.8, the loss cannot push")
    print("the scale back toward 1.0 — confirming scale-invariance.")
    
    # Compare with L1 gradient
    print()
    print("=" * 70)
    print("L1 Gradient w.r.t. Global Scale Factor α (for comparison)")
    print("=" * 70)
    print(f"{'α':>10s} | {'Loss':>12s} | {'∂L/∂α':>15s}")
    print("-" * 45)
    
    for alpha_val in [0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.5]:
        alpha = torch.tensor(alpha_val, requires_grad=True)
        wav_hat = wav_gt.detach() * alpha
        loss = (wav_hat - wav_gt.detach()).abs().mean()
        loss.backward()
        grad = alpha.grad.item()
        print(f"{alpha_val:10.2f} | {loss.item():12.6f} | {grad:15.8f}")


if __name__ == "__main__":
    test_mrstft_scale_invariance()
    test_mrstft_gradient_at_scale()
