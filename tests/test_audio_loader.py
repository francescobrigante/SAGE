# ===============================================================
# test_audio_loader.py
#
#   Unit tests for _torchaudio_load_safe and load_waveform.
#   Verifies SIGALRM timeout, handler restore, and channel/SR
#   conversion logic — no real audio files required.
# ===============================================================
import sys
import signal
import time
import pytest
import torch
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import ar_spectra.utils.audio as audio_mod
from ar_spectra.utils.audio import _torchaudio_load_safe, load_waveform

SIGALRM_AVAILABLE = hasattr(signal, "SIGALRM")


def _wav(channels: int = 2, samples: int = 44100, sr: int = 44100):
    return torch.zeros(channels, samples), sr


# ── T1: successful load returns tensor and sample rate ────────────────────────

def test_success_returns_wav():
    with patch("ar_spectra.utils.audio.torchaudio.load", return_value=_wav()) as mock:
        wav, sr = _torchaudio_load_safe("/fake/path.mp3")
    assert sr == 44100
    assert wav.shape == (2, 44100)
    # full-file defaults (frame_offset=0, num_frames=-1) reproduce the legacy call
    mock.assert_called_once_with("/fake/path.mp3", frame_offset=0, num_frames=-1, normalize=True)


# ── T2: timed-out load raises RuntimeError containing the path ────────────────

@pytest.mark.skipif(not SIGALRM_AVAILABLE, reason="SIGALRM not available on this platform")
def test_timeout_raises_runtime_error(monkeypatch):
    monkeypatch.setattr(audio_mod, "DEFAULT_AUDIO_LOAD_TIMEOUT", 1)

    def _slow(path, *args, **kwargs):
        time.sleep(4)
        return _wav()

    with patch("ar_spectra.utils.audio.torchaudio.load", side_effect=_slow):
        with pytest.raises(RuntimeError, match="/fake/path.mp3"):
            _torchaudio_load_safe("/fake/path.mp3")


# ── T3: SIGALRM handler and pending alarm are restored after success ──────────

@pytest.mark.skipif(not SIGALRM_AVAILABLE, reason="SIGALRM not available on this platform")
def test_signal_restored_after_success():
    prior_handler = signal.getsignal(signal.SIGALRM)

    with patch("ar_spectra.utils.audio.torchaudio.load", return_value=_wav()):
        _torchaudio_load_safe("/fake/path.mp3")

    assert signal.getsignal(signal.SIGALRM) is prior_handler
    # itimer value == 0 means no pending alarm
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0.0


# ── T4: SIGALRM handler and pending alarm are restored even after timeout ─────

@pytest.mark.skipif(not SIGALRM_AVAILABLE, reason="SIGALRM not available on this platform")
def test_signal_restored_after_timeout(monkeypatch):
    monkeypatch.setattr(audio_mod, "DEFAULT_AUDIO_LOAD_TIMEOUT", 1)
    prior_handler = signal.getsignal(signal.SIGALRM)

    def _slow(path, *args, **kwargs):
        time.sleep(4)
        return _wav()

    with patch("ar_spectra.utils.audio.torchaudio.load", side_effect=_slow):
        with pytest.raises(RuntimeError):
            _torchaudio_load_safe("/fake/path.mp3")

    assert signal.getsignal(signal.SIGALRM) is prior_handler
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0.0


# ── T5: load_waveform delegates to _torchaudio_load_safe (post-fix) ───────────
# Fails pre-fix (load_waveform calls torchaudio.load directly).
# Passes post-fix (load_waveform calls _torchaudio_load_safe).

def test_load_waveform_uses_safe_loader():
    with patch("ar_spectra.utils.audio._torchaudio_load_safe", return_value=_wav()) as mock:
        load_waveform("/fake/path.mp3", target_sample_rate=44100, expected_channels=2)
    # load_waveform forwards the (default, whole-file) windowing args through
    mock.assert_called_once_with("/fake/path.mp3", frame_offset=0, num_frames=-1)


# ── T6: load_waveform duplicates mono to stereo ───────────────────────────────

def test_load_waveform_mono_to_stereo():
    mono_wav, sr = _wav(channels=1, samples=22050)
    with patch("ar_spectra.utils.audio._torchaudio_load_safe", return_value=(mono_wav, sr)):
        wav, _, ch_mismatch, _ = load_waveform("/fake/path.mp3", target_sample_rate=sr, expected_channels=2)
    assert wav.shape[0] == 2
    assert ch_mismatch is True


# ── T7: load_waveform averages stereo to mono ─────────────────────────────────

def test_load_waveform_stereo_to_mono():
    stereo_wav, sr = _wav(channels=2, samples=22050)
    stereo_wav[0] = 1.0
    stereo_wav[1] = 3.0
    with patch("ar_spectra.utils.audio._torchaudio_load_safe", return_value=(stereo_wav, sr)):
        wav, _, ch_mismatch, _ = load_waveform("/fake/path.mp3", target_sample_rate=sr, expected_channels=1)
    assert wav.shape[0] == 1
    assert ch_mismatch is True
    assert torch.allclose(wav, torch.full((1, 22050), 2.0))


# ── T8: load_waveform resamples when SR mismatches ────────────────────────────

def test_load_waveform_resamples():
    wav_22k, _ = _wav(channels=2, samples=22050, sr=22050)
    with patch("ar_spectra.utils.audio._torchaudio_load_safe", return_value=(wav_22k, 22050)):
        wav, sr_out, _, sr_mismatch = load_waveform("/fake/path.mp3", target_sample_rate=44100, expected_channels=2)
    assert sr_out == 44100
    assert sr_mismatch is True
    # after 22050→44100 resampling, samples should double
    assert wav.shape[-1] == 44100
