import os
import torch
import logging
import numpy as np
from contextlib import contextmanager, redirect_stdout
import torchaudio
from pathlib import Path
from typing import Optional, List, Tuple
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from ar_spectra.utils.audio import load_waveform
from ar_spectra.utils.console import info, warn

class PairedEvalDataset(Dataset):
    """
    Dataset that returns aligned pairs of Target and Prediction waveforms.
    
    Args:
        target_dir: Directory containing reference audio.
        preds_dir: Directory containing reconstructed audio.
        sample_rate: Target sample rate for both waveforms.
        channels: Target channels (1 or 2).
        extensions: List of audio extensions to scan.
    """
    def __init__(
        self,
        target_dir: str | Path,
        preds_dir: str | Path,
        sample_rate: int = 44100,
        channels: int = 2,
        extensions: List[str] = [".wav", ".flac", ".mp3"],
        max_files: int = 0,
        fma_csv_path: Optional[str | Path] = None
    ):
        self.target_dir = Path(target_dir).expanduser().resolve()
        self.preds_dir = Path(preds_dir).expanduser().resolve()
        self.sample_rate = sample_rate
        self.channels = channels
        self.extensions = [ext.lower() for ext in extensions]
        self.fma_csv_path = Path(fma_csv_path) if fma_csv_path else None
        
        self.pairs = self._match_pairs()
        if max_files > 0:
            self.pairs = self.pairs[:max_files]
            
        if not self.pairs:
            raise RuntimeError(f"No matching pairs found between {self.target_dir} and {self.preds_dir}")
            
    def _match_pairs(self) -> List[Tuple[Path, Path]]:
        """Match files by stem name, filtering by FMA metadata if provided."""
        
        # 1. Load official test IDs if CSV provided
        test_ids = None
        if self.fma_csv_path and self.fma_csv_path.exists():
            try:
                import pandas as pd
                info(f"Filtering dataset to 'test' split using {self.fma_csv_path.name}...")
                tracks = pd.read_csv(self.fma_csv_path, index_col=0, header=[0, 1])
                # Filter strictly for 'test' split AND 'small' subset to get exactly 800 files
                test_condition = (tracks[('set', 'split')] == 'test') & (tracks[('set', 'subset')] == 'small')
                test_tracks = tracks[test_condition].index.tolist()
                test_ids = {f"{tid:06d}" for tid in test_tracks}
                info(f"Found {len(test_ids)} official test tracks in 'small' subset.")
            except Exception as e:
                warn(f"Failed to parse FMA CSV: {e}")

        # 2. Scan Targets with 'Shortest Path Win' logic to avoid recursive folders
        target_map = {}
        for f in self.target_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in self.extensions:
                # Basic skip for internal folders
                if any(p in f.parts for p in ["metrics", "embeddings", "cache"]):
                    continue
                
                stem = f.stem
                # Filter by test split if active
                if test_ids and stem not in test_ids:
                    continue
                
                # If we have duplicates, keep the one with the shortest path (least depth)
                if stem not in target_map or len(f.parts) < len(target_map[stem].parts):
                    target_map[stem] = f
                
        # 3. Scan Predictions and match with 'Shortest Path Win' logic for predictions too
        # We need to ensure 1:1 mapping even if runs/ has legacy recursive folders
        pred_map = {}
        for g in self.preds_dir.rglob("*"):
            if g.is_file() and g.suffix.lower() in self.extensions:
                if any(p in g.parts for p in ["metrics", "embeddings", "cache"]):
                    continue
                
                stem = g.stem
                if stem in target_map:
                    # Shortest path win for prediction file matches
                    if stem not in pred_map or len(g.parts) < len(pred_map[stem].parts):
                        pred_map[stem] = g
        
        pairs = [(target_map[s], p) for s, p in pred_map.items()]
        return sorted(pairs, key=lambda x: x[0].stem)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        target_path, pred_path = self.pairs[index]
        
        # Load and resample
        target_wav, _, _, _ = load_waveform(
            target_path, 
            target_sample_rate=self.sample_rate, 
            expected_channels=self.channels
        )
        pred_wav, _, _, _ = load_waveform(
            pred_path, 
            target_sample_rate=self.sample_rate, 
            expected_channels=self.channels
        )
        
        # Ensure they are contiguous
        return target_wav.contiguous(), pred_wav.contiguous(), target_path.stem

def collate_paired_eval(batch):
    """
    Collate function that pads waveforms to the max length in the batch.
    Returns: (target_batch, pred_batch, stems)
    """
    targets, preds, stems = zip(*batch)
    
    # Track original lengths for each item in batch
    # (Though usually prediction is matched to target length anyway)
    
    # For batching, we pad with zeros to the longest sample size in this batch
    max_len = max([t.shape[-1] for t in targets] + [p.shape[-1] for p in preds])
    
    padded_targets = []
    padded_preds = []
    
    for t, p in zip(targets, preds):
        # Pad target
        t_pad = torch.nn.functional.pad(t, (0, max_len - t.shape[-1]))
        padded_targets.append(t_pad)
        # Pad pred
        p_pad = torch.nn.functional.pad(p, (0, max_len - p.shape[-1]))
        padded_preds.append(p_pad)
        
    return torch.stack(padded_targets), torch.stack(padded_preds), stems

def batch_align(target: torch.Tensor, pred: torch.Tensor, sr: int, max_shift_seconds: float = 1.0):
    """
    GPU-accelerated batched alignment using cross-correlation via FFT.
    Args:
        target: [B, C, T] reference
        pred:   [B, C, T] prediction to shift
    Returns:
        (aligned_target, aligned_pred, lags)
    """
    B, C, T = target.shape
    device = target.device
    
    # 1. Prepare for FFT: average channels and remove DC
    t_m = target.mean(dim=1)  # [B, T]
    p_m = pred.mean(dim=1)    # [B, T]
    
    t_m = t_m - t_m.mean(dim=-1, keepdim=True)
    p_m = p_m - p_m.mean(dim=-1, keepdim=True)
    
    # 2. Compute cross-correlation via FFT
    # Length for full correlation: 2T-1
    L = 2 * T - 1
    n_fft = 1
    while n_fft < L:
        n_fft *= 2
        
    t_flip = torch.flip(t_m, dims=[-1])
    
    T_FFT = torch.fft.rfft(t_flip, n=n_fft)
    P_FFT = torch.fft.rfft(p_m, n=n_fft)
    
    corr = torch.fft.irfft(P_FFT * T_FFT, n=n_fft)
    corr = corr[:, :L] # [B, L]
    
    # 3. Find lags
    # lags range from -T+1 to T-1
    lags = torch.arange(-T + 1, T, device=device)
    
    # Window the correlation to max_shift
    max_shift = int(max_shift_seconds * sr)
    mask = (lags >= -max_shift) & (lags <= max_shift)
    
    corr_masked = corr[:, mask]
    lags_masked = lags[mask]
    
    best_idx = torch.argmax(corr_masked, dim=-1) # [B]
    best_lags = lags_masked[best_idx] # [B]
    
    # 4. Apply shifts
    # We apply shifts individually. Since we are in a batch and tracks might have 
    # different lags, we'll return the shifted tensors.
    # Note: Shifted parts are zero-filled or truncated.
    
    # For SI-SDR/CDPAM, we precisely align by cropping or shifting.
    # To keep the batch structure [B, C, T_new], we'll return a batch 
    # where each pair is aligned and the rest is zeroed.
    
    aligned_pred = torch.zeros_like(pred)
    aligned_target = torch.zeros_like(target)
    
    for i in range(B):
        lag = int(best_lags[i].item())
        x = target[i]
        y = pred[i]
        
        if lag > 0: # y is delayed
            y_aligned = y[..., lag:]
            x_aligned = x[..., :y_aligned.shape[-1]]
        elif lag < 0: # y is advanced
            lag_abs = abs(lag)
            x_aligned = x[..., lag_abs:]
            y_aligned = y[..., :x_aligned.shape[-1]]
        else:
            x_aligned, y_aligned = x, y
            
        # Write back to aligned tensors (left-aligned)
        cur_len = x_aligned.shape[-1]
        aligned_target[i, :, :cur_len] = x_aligned
        aligned_pred[i, :, :cur_len] = y_aligned
        
    return aligned_target, aligned_pred, best_lags

def atomic_save_npy(path: Path, data: np.ndarray):
    """Save numpy array atomically to prevent race conditions."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "wb") as f:
        np.save(f, data)
    os.replace(tmp_path, path)

@contextmanager
def silence_output():
    """Context manager to silence stdout and logging."""
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull):
            # Also silence logging
            logger = logging.getLogger()
            old_level = logger.level
            logger.setLevel(logging.ERROR)
            try:
                yield
            finally:
                logger.setLevel(old_level)

def batch_embed_files(fad, files: List[Path], batch_size: int = 16, cache_dir: Path = None, subfolder: Optional[str] = None):
    """
    Compute and cache embeddings for a list of files using GPU inference.
    For CLAP/MERT models: true batched inference (B files × chunk positions).
    For other models (VGGish): per-file inference via fadtk's get_embedding.
    """
    model_name = fad.ml.name
    sr = fad.ml.sr
    chunk_size = 10 * sr        # 10 s window
    hop_size = sr               # 1 s hop

    # Filter already-cached files
    to_embed = []
    from fadtk.fad import get_cache_embedding_path
    for f in files:
        if not cache_dir:
            cache_path = get_cache_embedding_path(model_name, f)
        else:
            if subfolder:
                cache_path = cache_dir / model_name / subfolder / f"{f.stem}.npy"
            else:
                cache_path = cache_dir / model_name / f"{f.stem}.npy"
        
        if not cache_path.exists():
            to_embed.append((f, cache_path))

    if not to_embed:
        info(f"All {len(files)} files are already in cache for {model_name}.")
        return

    info(f"Computing embeddings for {len(to_embed)} files in batches of {batch_size} [{model_name}]...")

    if fad.ml.model is None:
        with silence_output():
            fad.ml.load_model()
        info(f"Model {model_name} loaded successfully.")
    device = fad.ml.device

    # Model detection
    try:
        from fadtk.model_loader import CLAPLaionModel as _CLAPLaionModel
        is_clap = isinstance(fad.ml, _CLAPLaionModel)
    except ImportError:
        is_clap = hasattr(fad.ml.model, 'get_audio_embedding_from_data')

    try:
        from fadtk.model_loader import MERTModel as _MERTModel
        is_mert = isinstance(fad.ml, _MERTModel)
    except ImportError:
        is_mert = "mert" in model_name.lower()
        
    if is_mert and hasattr(fad.ml.model.encoder, "pos_conv_embed"):
        m_conv = fad.ml.model.encoder.pos_conv_embed.conv
        # Monkey-patch PyTorch >= 2.2 parametrization missing keys:
        if hasattr(m_conv, "parametrizations"):
            try:
                from huggingface_hub import hf_hub_download
                ckpt_path = hf_hub_download(fad.ml.huggingface_id, 'pytorch_model.bin')
                state_dict = torch.load(ckpt_path, map_location='cpu')
                if "encoder.pos_conv_embed.conv.weight_g" in state_dict:
                    with torch.no_grad():
                        m_conv.parametrizations.weight.original0.copy_(state_dict["encoder.pos_conv_embed.conv.weight_g"])
                        m_conv.parametrizations.weight.original1.copy_(state_dict["encoder.pos_conv_embed.conv.weight_v"])
                    info(f"MERT pos_conv_embed weights monkey-patched for PyTorch >= 2.x compatibility.")
            except Exception as e:
                warn(f"Failed to monkey-patch MERT pos_conv_embed: {e}")

    if not is_clap and not is_mert:
        # Generic per-file path for VGGish and any future model
        for f, cp in tqdm(to_embed, desc=f"Embedding {model_name}"):
            if cp.exists():
                continue
            try:
                wav = fad.load_audio(f)
                emb = fad.ml.get_embedding(wav)
                if isinstance(emb, torch.Tensor):
                    emb = emb.cpu().detach().numpy()
                atomic_save_npy(cp, np.atleast_2d(emb).astype(np.float16))
            except Exception as e:
                warn(f"Failed to embed {f}: {e}")
        return

    # --- MERT-specific batched path ---
    if is_mert:
        def _load_and_chunk_mert(f: Path) -> list[torch.Tensor]:
            """Load audio, resample to 24k if needed, split into chunks."""
            import torchaudio.transforms as T
            wav, sr = torchaudio.load(f)
            if sr != 24000:
                wav = T.Resample(sr, 24000)(wav)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            
            # MERT expects normalized audio [-1, 1] - usually handled by processor
            # but we chunk it first.
            T_target = wav.shape[1]
            chunks = []
            chunk_len = 5 * 24000 # 5s chunks for MERT is typical
            hop_len = 1 * 24000
            for start in range(0, T_target, hop_len):
                chunk = wav[:, start:start + chunk_len]
                if chunk.shape[1] < chunk_len:
                    chunk = torch.nn.functional.pad(chunk, (0, chunk_len - chunk.shape[1]))
                chunks.append(chunk[0]) # (L,)
            return chunks

        pbar = tqdm(range(0, len(to_embed), batch_size), desc=f"Embedding {model_name} (batched)",
                    total=(len(to_embed) + batch_size - 1) // batch_size)
        
        for batch_start in pbar:
            batch = to_embed[batch_start:batch_start + batch_size]
            loaded = []
            for local_i, (f, cp) in enumerate(batch):
                try:
                    loaded.append((local_i, _load_and_chunk_mert(f)))
                except Exception as e:
                    warn(f"Failed to load {f}: {e}")
            
            if not loaded: continue
            
            n_chunks = max(len(ch) for _, ch in loaded)
            file_embs = {local_i: [] for local_i, _ in loaded}
            
            with torch.no_grad():
                for chunk_idx in range(n_chunks):
                    chunk_batch, active = [], []
                    for local_i, chunks in loaded:
                        if chunk_idx < len(chunks):
                            chunk_batch.append(chunks[chunk_idx])
                            active.append(local_i)
                    if not chunk_batch: continue
                    
                    # Process via MERT processor + model
                    # processor expects list of numpy/tensors
                    # converting to numpy often avoids dimension issues in some transformers versions
                    chunk_batch_np = [t.cpu().numpy() for t in chunk_batch]
                    inputs = fad.ml.processor(chunk_batch_np, sampling_rate=24000, return_tensors="pt", padding=True).to(device)
                    
                    # Ensure all inputs are correctly shaped (B, T)
                    for k in inputs:
                        if isinstance(inputs[k], torch.Tensor):
                            # Squeeze extra leading dimensions if batch size is hidden inside
                            # We want to reach (B, ...) where B = len(chunk_batch)
                            while inputs[k].ndim > 2 and inputs[k].shape[0] == 1:
                                inputs[k] = inputs[k].squeeze(0)
                            # If it's still [1, B, L] or similar, one more squeeze might be needed
                            # but len(chunk_batch) should be the first dim.
                            if inputs[k].shape[0] != len(chunk_batch) and inputs[k].ndim > 1:
                                if inputs[k].shape[1] == len(chunk_batch):
                                    inputs[k] = inputs[k].squeeze(0)

                    outputs = fad.ml.model(**inputs, output_hidden_states=True)
                    
                    # Use specified layer (default 12)
                    layer_idx = getattr(fad.ml, 'layer', 12)
                    hidden_states = outputs.hidden_states[layer_idx]
                    # Mean pooling over time dimension
                    emb = hidden_states.mean(dim=1) # (B, D)
                    
                    for j, local_i in enumerate(active):
                        file_embs[local_i].append(emb[j].unsqueeze(0).cpu())
            
            for local_i, _ in loaded:
                if not file_embs[local_i]: continue
                f, cp = batch[local_i]
                final_emb = torch.cat(file_embs[local_i], dim=0).numpy().astype(np.float16)
                atomic_save_npy(cp, final_emb)
        return

    # --- CLAP-specific batched path ---
    # Note: int16_to_float32/float32_to_int16 are module-level functions in
    # laion_clap.training.data — NOT methods on CLAP_Module.
    from laion_clap.training.data import int16_to_float32 as _i16tof32, \
                                         float32_to_int16 as _f32toi16

    def _load_and_chunk(f: Path) -> list[np.ndarray]:
        """Load audio, apply CLAP quantization, split into padded 10-s chunks."""
        wav = fad.load_audio(f).reshape(1, -1)      # (1, T) float32 at sr
        wav = _i16tof32(_f32toi16(wav))
        T = wav.shape[1]
        chunks = []
        for start in range(0, T, hop_size):
            chunk = wav[:, start:start + chunk_size]  # (1, L)
            if chunk.shape[1] < chunk_size:
                chunk = np.pad(chunk, ((0, 0), (0, chunk_size - chunk.shape[1])))
            chunks.append(chunk[0])  # (chunk_size,)
        return chunks

    pbar = tqdm(range(0, len(to_embed), batch_size), desc=f"Embedding {model_name}",
                total=(len(to_embed) + batch_size - 1) // batch_size)
    for batch_start in pbar:
        batch = to_embed[batch_start:batch_start + batch_size]

        # Load & chunk all files in this batch
        loaded: list[tuple[int, list[np.ndarray]]] = []  # (local_idx, chunks)
        for local_i, (f, cp) in enumerate(batch):
            if cp.exists():
                continue  # race protection
            try:
                loaded.append((local_i, _load_and_chunk(f)))
            except Exception as e:
                warn(f"Failed to load {f}: {e}")

        if not loaded:
            continue

        n_chunks = max(len(ch) for _, ch in loaded)
        file_embs: dict[int, list] = {local_i: [] for local_i, _ in loaded}

        # Batched forward: one model call per chunk position
        with torch.no_grad():
            for chunk_idx in range(n_chunks):
                chunk_batch, active = [], []
                for local_i, chunks in loaded:
                    if chunk_idx < len(chunks):
                        chunk_batch.append(chunks[chunk_idx])
                        active.append(local_i)
                if not chunk_batch:
                    continue
                tensor = torch.from_numpy(np.stack(chunk_batch, axis=0)).float().to(device)
                # tensor: (B, chunk_size) — model supports arbitrary batch size
                embs = fad.ml.model.get_audio_embedding_from_data(x=tensor, use_tensor=True)
                for j, local_i in enumerate(active):
                    file_embs[local_i].append(embs[j].unsqueeze(0))  # (1, D)

        # Save per-file embeddings
        for local_i, chunks in loaded:
            if not file_embs[local_i]:
                continue
            f, cp = batch[local_i]
            emb = torch.cat(file_embs[local_i], dim=0).cpu().detach().numpy().astype(np.float16)
            atomic_save_npy(cp, emb)

def load_pooled_embedding(path: Path) -> np.ndarray:
    """Load cached .npy embedding, mean-pool over time axis, return shape (D,)."""
    emb = np.atleast_2d(np.load(path))  # (T, D) or already (1, D)
    return emb.mean(axis=0)            # (D,)
