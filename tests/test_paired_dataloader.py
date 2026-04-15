import os
import sys
import torch
import time
import shutil
from pathlib import Path
from torch.utils.data import DataLoader

# Add project root and evaluation/ to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))

from eval_dataloader import PairedEvalDataset, collate_paired_eval, batch_align
from config import DATA_PATH, RUNS_DIR
from ar_spectra.utils.console import ok, info, warn, err

def test_alignment():
    info("Testing Batched GPU Alignment with synthetic lag...")
    
    B, C, T = 4, 1, 44100 * 2  # 2 seconds, mono for simplicity
    sr = 44100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a random signal
    target = torch.randn(B, C, T).to(device)
    
    # Create shifted versions (lags in samples)
    true_lags = torch.tensor([100, -50, 200, 0], device=device)
    pred = torch.zeros_like(target)
    
    for i in range(B):
        lag = int(true_lags[i].item())
        if lag >= 0:
            # pred is delayed: pred[t] = target[t - lag]
            pred[i, :, lag:] = target[i, :, :T-lag]
        else:
            # pred is advanced: pred[t] = target[t - lag] (lag < 0)
            lag_abs = abs(lag)
            pred[i, :, :T-lag_abs] = target[i, :, lag_abs:]
            
    # Run batched alignment
    aligned_t, aligned_p, detected_lags = batch_align(target, pred, sr)
    
    info(f"True lags:     {true_lags.cpu().tolist()}")
    info(f"Detected lags: {detected_lags.cpu().tolist()}")
    
    error = torch.abs(true_lags - detected_lags).max().item()
    if error == 0:
        ok("Alignment test passed: Lags detected perfectly.")
    else:
        err(f"Alignment test failed: Max lag error = {error}")

def test_dataloader_speed():
    info("Benchmarking PairedEvalDataLoader speed...")
    
    # Use real data paths if available
    target_dir = DATA_PATH
    # For testing, we use the same dir as predictions just to see loading speed
    # (since we don't have a full prediction run yet for 32 files)
    preds_dir = DATA_PATH 
    
    try:
        dataset = PairedEvalDataset(target_dir, preds_dir, sample_rate=44100, channels=2)
    except Exception as e:
        warn(f"Could not initialize with real data: {e}")
        return

    loader = DataLoader(
        dataset, 
        batch_size=16, 
        shuffle=False, 
        num_workers=8, 
        collate_fn=collate_paired_eval
    )
    
    info(f"Dataset size: {len(dataset)} pairs")
    num_batches = 2 # Test 32 files (2 batches of 16)
    
    start_time = time.time()
    count = 0
    for i, (t_batch, p_batch, stems) in enumerate(loader):
        count += t_batch.shape[0]
        info(f"Batch {i+1} loaded: {t_batch.shape}, {len(stems)} stems")
        if i + 1 >= num_batches:
            break
            
    elapsed = time.time() - start_time
    ok(f"Loaded {count} files in {elapsed:.2f}s ({count/elapsed:.2f} files/s)")

def main():
    info("Starting Dataloader Verification Phase")
    print("-" * 40)
    
    test_alignment()
    print("-" * 40)
    
    test_dataloader_speed()
    print("-" * 40)
    
    ok("Dataloader verification complete.")

if __name__ == "__main__":
    main()
