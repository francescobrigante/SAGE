"""
Test STFT → encode/decode → iSTFT round-trip amplitude fidelity.

Tests each stage of the pipeline independently to find where amplitude
is lost:

1. Pure STFT→iSTFT round-trip (no model)
2. STFT→crop1025to1024→addNyquist→iSTFT (freq crop only)
3. STFT→pre_transform→inverse_pre_transform→iSTFT
4. STFT→encoder→decoder→iSTFT (identity/untrained model)
5. Full pipeline with a random-init model
"""
import torch
import sys

import torchaudio


def test_pure_stft_roundtrip():
    """Test 1: Pure torch.stft → torch.istft"""
    print("\n" + "=" * 70)
    print("Test 1: Pure STFT → iSTFT round-trip")
    print("=" * 70)
    
    torch.manual_seed(42)
    B, C, T = 2, 2, 65024  # (128-1)*512 = 65024
    wav = torch.randn(B, C, T) * 0.5
    
    n_fft, hop, win = 2048, 512, 2048
    window = torch.hann_window(win)
    
    # Flatten to (B*C, T)
    wav_flat = wav.reshape(B * C, T)
    spec = torch.stft(wav_flat, n_fft=n_fft, hop_length=hop, win_length=win,
                       window=window, center=True, normalized=False,
                       onesided=True, return_complex=True)
    print(f"  STFT shape: {spec.shape}")  # should be (4, 1025, 128)
    
    wav_recon = torch.istft(spec, n_fft=n_fft, hop_length=hop, win_length=win,
                            window=window, center=True, normalized=False,
                            onesided=True, return_complex=False, length=T)
    wav_recon = wav_recon.reshape(B, C, T)
    
    ratio = wav_recon.abs().mean() / wav.abs().mean()
    max_error = (wav_recon - wav).abs().max()
    print(f"  Amplitude ratio (recon/orig): {ratio:.8f}")
    print(f"  Max abs error: {max_error:.2e}")
    assert abs(ratio - 1.0) < 1e-5, f"STFT roundtrip amplitude mismatch: {ratio}"
    print("  ✓ PASS")


def test_freq_crop_roundtrip():
    """Test 2: STFT → crop 1025→1024 → pad zeros → iSTFT"""
    print("\n" + "=" * 70)
    print("Test 2: STFT → crop 1025→1024 → pad Nyquist → iSTFT")
    print("=" * 70)
    
    torch.manual_seed(42)
    B, C, T = 2, 2, 65024
    wav = torch.randn(B, C, T) * 0.5
    
    n_fft, hop, win = 2048, 512, 2048
    window = torch.hann_window(win)
    
    wav_flat = wav.reshape(B * C, T)
    spec = torch.stft(wav_flat, n_fft=n_fft, hop_length=hop, win_length=win,
                       window=window, center=True, normalized=False,
                       onesided=True, return_complex=True)
    
    # Drop Nyquist bin (1025 → 1024) and re-add as zeros
    spec_cropped = spec[:, :1024, :]
    nyq = torch.zeros(spec_cropped.shape[0], 1, spec_cropped.shape[2], 
                       dtype=spec.dtype, device=spec.device)
    spec_padded = torch.cat([spec_cropped, nyq], dim=1)
    
    wav_recon = torch.istft(spec_padded, n_fft=n_fft, hop_length=hop, win_length=win,
                            window=window, center=True, normalized=False,
                            onesided=True, return_complex=False, length=T)
    wav_recon = wav_recon.reshape(B, C, T)
    
    ratio = wav_recon.abs().mean() / wav.abs().mean()
    max_error = (wav_recon - wav).abs().max()
    rms_orig = wav.pow(2).mean().sqrt()
    rms_recon = wav_recon.pow(2).mean().sqrt()
    print(f"  Amplitude ratio (recon/orig): {ratio:.8f}")
    print(f"  RMS ratio: {rms_recon / rms_orig:.8f}")
    print(f"  Max abs error: {max_error:.2e}")
    print(f"  ✓ Nyquist crop causes {(1 - ratio)*100:.4f}% amplitude loss")


def test_torchaudio_spectrogram_vs_torch_stft():
    """Test 3: Compare dataloader STFT (torchaudio.transforms.Spectrogram)
    vs model STFT (torch.stft)"""
    print("\n" + "=" * 70)
    print("Test 3: torchaudio.transforms.Spectrogram vs torch.stft")
    print("=" * 70)
    
    torch.manual_seed(42)
    C, T = 2, 65024
    wav = torch.randn(C, T) * 0.5
    
    n_fft, hop, win = 2048, 512, 2048
    
    # torchaudio Spectrogram (as used in dataloader)
    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, hop_length=hop, win_length=win,
        window_fn=torch.hann_window, power=None,
        center=True, pad_mode="reflect", normalized=False,
    )
    spec_ta = spec_transform(wav)  # (C, F, T)
    
    # torch.stft (as used in autoencoder.stft())
    window = torch.hann_window(win)
    wav_flat = wav.reshape(C, T)
    spec_torch = torch.stft(wav_flat, n_fft=n_fft, hop_length=hop, win_length=win,
                             window=window, center=True, normalized=False,
                             onesided=True, return_complex=True)
    
    diff = (spec_ta - spec_torch).abs().max()
    mag_ratio = spec_ta.abs().mean() / spec_torch.abs().mean()
    
    print(f"  torchaudio spec shape: {spec_ta.shape}")
    print(f"  torch.stft spec shape: {spec_torch.shape}")
    print(f"  Max abs difference: {diff:.2e}")
    print(f"  Magnitude ratio (torchaudio/torch.stft): {mag_ratio:.8f}")
    
    if diff < 1e-5:
        print("  ✓ PASS - identical")
    else:
        print(f"  ✗ FAIL - difference: {diff}")


def test_pre_transform_roundtrip():
    """Test 4: Pre-transform → inverse pre-transform on complex spectrogram"""
    print("\n" + "=" * 70)
    print("Test 4: PowerMagnitudeTransform round-trip")
    print("=" * 70)
    
    from sage.nn.pre_transform import PowerMagnitudeTransform
    
    torch.manual_seed(42)
    B, C, F, T = 2, 2, 1025, 128
    
    # Create a realistic complex spectrogram
    mag = torch.rand(B, C, F, T) * 10.0 + 0.01  # positive magnitudes
    phase = torch.rand(B, C, F, T) * 2 * 3.14159 - 3.14159
    spec = torch.polar(mag, phase)  # complex
    
    pt = PowerMagnitudeTransform(alpha=0.65, beta=0.35, eps=1e-8)
    
    transformed = pt.transform(spec)
    restored = pt.inverse(transformed)
    
    ratio = restored.abs().mean() / spec.abs().mean()
    max_error = (restored - spec).abs().max()
    
    print(f"  Original mean magnitude: {spec.abs().mean():.6f}")
    print(f"  Restored mean magnitude: {restored.abs().mean():.6f}")
    print(f"  Amplitude ratio (restored/orig): {ratio:.8f}")
    print(f"  Max abs error: {max_error:.2e}")
    
    if abs(ratio - 1.0) < 1e-4:
        print("  ✓ PASS")
    else:
        print(f"  ✗ FAIL - {(1-ratio)*100:.4f}% amplitude loss")


def test_full_autoencoder_stft_istft():
    """Test 5: Full SAGEAutoencoder.stft() → SAGEAutoencoder.istft() with freq crop"""
    print("\n" + "=" * 70)
    print("Test 5: SAGEAutoencoder.stft() → istft() (no model, just STFT/iSTFT)")
    print("=" * 70)
    
    from sage.model.autoencoder import SAGEAutoencoder
    import torch.nn as nn
    
    # Create a minimal autoencoder just for STFT/iSTFT
    class DummyEncoder(nn.Module):
        def forward(self, x):
            return x, {}
    class DummyDecoder(nn.Module):
        def forward(self, x, encoder_info=None):
            return x
    
    ae = SAGEAutoencoder(encoder=DummyEncoder(), decoder=DummyDecoder())
    ae.set_stft_config({"n_fft": 2048, "hop_length": 512, "win_length": 2048})
    
    torch.manual_seed(42)
    B, C, T = 2, 2, 65024
    wav = torch.randn(B, C, T) * 0.5
    
    # STFT
    spec = ae.stft(wav)
    print(f"  STFT output: {spec.shape}, dtype={spec.dtype}")
    
    # Simulate what Swin encoder does: crop 1025 → 1024
    spec_cropped = spec[..., :1024, :]
    print(f"  After freq crop: {spec_cropped.shape}")
    
    # iSTFT with _maybe_add_nyquist
    wav_recon = ae.istft(spec_cropped, target_length=T)
    print(f"  Reconstructed wav: {wav_recon.shape}")
    
    ratio = wav_recon.abs().mean() / wav.abs().mean()
    rms_ratio = wav_recon.pow(2).mean().sqrt() / wav.pow(2).mean().sqrt()
    print(f"  Amplitude ratio: {ratio:.8f}")
    print(f"  RMS ratio: {rms_ratio:.8f}")
    
    # Now test without crop (full 1025 bins)
    wav_recon_full = ae.istft(spec, target_length=T)
    ratio_full = wav_recon_full.abs().mean() / wav.abs().mean()
    print(f"  Amplitude ratio (no crop): {ratio_full:.8f}")
    
    print(f"\n  Freq crop amplitude loss: {(1-ratio)*100:.4f}%")
    print(f"  Full roundtrip loss: {(1-ratio_full)*100:.6f}%")


def test_pack_unpack_complex():
    """Test 6: _pack_complex → _unpack_complex round-trip"""
    print("\n" + "=" * 70)
    print("Test 6: _pack_complex → _unpack_complex round-trip")
    print("=" * 70)
    
    from sage.model.autoencoder import SAGEAutoencoder
    import torch.nn as nn
    
    class DummyEncoder(nn.Module):
        def forward(self, x):
            return x, {}
    class DummyDecoder(nn.Module):
        def forward(self, x, encoder_info=None):
            return x
    
    ae = SAGEAutoencoder(encoder=DummyEncoder(), decoder=DummyDecoder())
    
    torch.manual_seed(42)
    B, C, F, T = 2, 2, 1024, 128
    spec = torch.randn(B, C, F, T, dtype=torch.float32) + 1j * torch.randn(B, C, F, T, dtype=torch.float32)
    
    packed = ae._pack_complex(spec)  # (B, 2C, F, T) 
    unpacked = ae._unpack_complex(packed)  # (B, C, F, T)
    
    diff = (unpacked - spec).abs().max()
    ratio = unpacked.abs().mean() / spec.abs().mean()
    
    print(f"  Original: {spec.shape}, dtype={spec.dtype}")
    print(f"  Packed: {packed.shape}, dtype={packed.dtype}")
    print(f"  Unpacked: {unpacked.shape}, dtype={unpacked.dtype}")
    print(f"  Max error: {diff:.2e}")
    print(f"  Amplitude ratio: {ratio:.8f}")
    
    if diff < 1e-6:
        print("  ✓ PASS")
    else:
        print(f"  ✗ FAIL")


def test_encode_decode_amplitude_with_swin():
    """Test 7: Full encode→decode amplitude with actual Swin model (random init)"""
    print("\n" + "=" * 70)
    print("Test 7: Full Swin encode→decode with random weights")
    print("=" * 70)
    
    from sage.model.autoencoder import SAGEAutoencoder
    from sage.model.encoder import SAGEEncoder
    from sage.model.decoder import SAGEDecoder
    from sage.nn.complex.bottleneck import ComplexVAEBottleneck
    from sage.nn.pre_transform import PowerMagnitudeTransform
    
    torch.manual_seed(42)
    
    encoder = SAGEEncoder(
        in_channels=2, embed_dim=64, depths=[2, 2, 6, 2],
        num_heads=[4, 8, 16, 32], window_size=8, patch_size=4,
        dimension=24,  # 3 * 8
        is_complex=True, complex_activation="ComplexGELU1d",
    )
    
    decoder = SAGEDecoder(
        channels=8, in_channels=2, embed_dim=64,
        depths=[2, 6, 2, 2], num_heads=[32, 16, 8, 4],
        window_size=8, patch_size=4,
        is_complex=True, complex_activation="ComplexGELU1d",
    )
    
    bottleneck = ComplexVAEBottleneck(
        apply_cholesky_constraints=False,
        apply_spectral_parameterization=False,
        proper=False,
    )
    
    ae = SAGEAutoencoder(
        encoder=encoder, decoder=decoder, bottleneck=bottleneck,
        pre_transform={"type": "power_norm", "apply_target": True,
                       "apply_inverse": True, "config": {"alpha": 0.65, "beta": 0.35}},
    )
    ae.set_stft_config({"n_fft": 2048, "hop_length": 512, "win_length": 2048})
    ae.eval()
    
    B, C, T = 1, 2, 65024
    wav = torch.randn(B, C, T) * 0.3
    
    with torch.no_grad():
        spec = ae.stft(wav)
        print(f"  STFT: {spec.shape}")
        
        # Encode (includes pre_transform)
        latents = ae.encode(spec)
        print(f"  Latents: {latents.shape}")
        
        # Decode (includes inverse pre_transform)
        spec_recon = ae.decode(latents, apply_inverse=True)
        print(f"  Decoded spec: {spec_recon.shape}")
        
        # iSTFT
        wav_recon = ae.istft(spec_recon, target_length=T)
        print(f"  Recon wav: {wav_recon.shape}")
    
    ratio = wav_recon.abs().mean() / wav.abs().mean()
    rms_ratio = wav_recon.pow(2).mean().sqrt() / wav.pow(2).mean().sqrt()
    print(f"\n  Amplitude ratio: {ratio:.6f}")
    print(f"  RMS ratio: {rms_ratio:.6f}")
    
    if ratio < 0.95 or ratio > 1.05:
        print(f"  ⚠ Amplitude mismatch with random init: {ratio:.4f}")
        print(f"    This is EXPECTED with random weights.")
    else:
        print("  ✓ Within 5%")


if __name__ == "__main__":
    test_pure_stft_roundtrip()
    test_freq_crop_roundtrip()
    test_torchaudio_spectrogram_vs_torch_stft()
    test_pre_transform_roundtrip()
    test_full_autoencoder_stft_istft()
    test_pack_unpack_complex()
    test_encode_decode_amplitude_with_swin()
