import os
import json
import argparse
import resource
from pathlib import Path

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from ar_spectra.dataset import OnTheFlySTFTDataset  # ensures module import and side effects if any
from ar_spectra.models.autoencoder import instantiate_from_spec


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def max_pinnable_mb() -> int:
    if not torch.cuda.is_available():
        return 0
    mb_list = [8, 16, 24, 32, 40, 48, 56, 64, 96, 128, 192, 256, 384, 512]
    last_ok = 0
    for mb in mb_list:
        try:
            x = torch.empty((mb * 1024 * 1024) // 4, dtype=torch.float32)
            x.pin_memory()
            last_ok = mb
        except Exception:
            break
    return last_ok


def print_env_info():
    import torchaudio
    print("CUDA_VISIBLE_DEVICES =", os.getenv("CUDA_VISIBLE_DEVICES"))
    print("torch:", torch.__version__, "| cuda:", torch.version.cuda, "| available:", torch.cuda.is_available())
    print("torchaudio:", torchaudio.__version__)
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    print(f"RLIMIT_MEMLOCK soft={soft} hard={hard} (KB reported by 'ulimit -l')")
    if torch.cuda.is_available():
        print("cuda device_count:", torch.cuda.device_count())
        torch.cuda.set_device(0)
        print("current_device:", torch.cuda.current_device())


def pin_smoke_test():
    if not torch.cuda.is_available():
        print("[SMOKE] CUDA not available, skipping pin test.")
        return

    def try_bytes(n_bytes: int) -> bool:
        num_floats = max(1, n_bytes // 4)
        try:
            x = torch.empty(num_floats, dtype=torch.float32)
            x.pin_memory()
            return True
        except Exception as e:
            print(f"[SMOKE] pin_memory({n_bytes/1024/1024:.2f} MB) FAILED:", repr(e))
            return False

    for mb in [1, 4, 8, 16, 32, 64, 128]:
        ok = try_bytes(mb * 1024 * 1024)
        print(f"[SMOKE] {mb} MB pinnable? {ok}")

    try:
        x = torch.randn(8, 4, 257, 512)
        x.pin_memory()
        print("[SMOKE] pin_memory ~16.8MB OK")
    except Exception as e:
        print("[SMOKE] pin_memory ~16.8MB FAILED:", repr(e))

    try:
        ds = TensorDataset(torch.randn(2, 4, 257, 512), torch.randn(2, 2, 65408))
        dl = DataLoader(ds, batch_size=2, num_workers=0, pin_memory=True)
        for _ in dl:
            break
        print("[SMOKE] DataLoader pin_memory OK on synthetic dataset.")
    except Exception as e:
        print("[SMOKE] DataLoader pin_memory FAILED:", repr(e))


def collate_stft(batch):
    for k, item in enumerate(batch):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError(f"Batch item {k} is not a pair (S, W): {type(item)}")
        S, W = item
        if not isinstance(S, torch.Tensor) or not isinstance(W, torch.Tensor):
            raise TypeError(f"Batch item {k} contains non-Tensor: S={type(S)}, W={type(W)}")
        if S.is_cuda or W.is_cuda:
            raise RuntimeError(f"Batch item {k} contains CUDA tensors. pin_memory requires CPU. "
                               f"S.device={S.device}, W.device={W.device}")
        if not S.is_contiguous():
            S = S.contiguous()
        if not W.is_contiguous():
            W = W.contiguous()
        batch[k] = (S, W)

    Ss, wavs = zip(*batch)
    s0 = Ss[0].shape
    w0 = wavs[0].shape
    assert all(x.shape == s0 for x in Ss), f"STFT shapes differ: {[x.shape for x in Ss]}"
    assert all(x.shape == w0 for x in wavs), f"Wave shapes differ: {[x.shape for x in wavs]}"
    return torch.stack(Ss, 0), torch.stack(wavs, 0)


def main():
    parser = argparse.ArgumentParser(description="Dataset/DataLoader pinning debug utility")
    parser.add_argument("--config", type=str,
                        default="/home/cerovaz/repos/ICML/Eulero_BackBone/ar_spectra/config/experiments/SEANet_STFT.json",
                        help="Path to experiment JSON config")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--pin-memory", type=str, choices=["auto", "true", "false"], default="auto",
                        help="Enable pinned memory: auto uses a heuristic vs. pinnable MB")
    args = parser.parse_args()

    print_env_info()

    cfg = load_json(args.config)
    bs_cfg = int(cfg.get("train_dataloader", {}).get("batch_size", 8))
    batch_size = args.batch_size if args.batch_size is not None else bs_cfg
    print(f"Using batch size: {batch_size}")

    pin_smoke_test()

    print("Instantiating dataset...")
    dataset = instantiate_from_spec(cfg["train_dataset"])
    print(f"Dataset ready. Found {len(dataset)} files.")

    try:
        s0, w0 = dataset[0]
        print(f"First sample - devices S={s0.device}, W={w0.device}, "
              f"dtypes S={s0.dtype}, W={w0.dtype}, shapes S={tuple(s0.shape)}, W={tuple(w0.shape)}")
    except Exception as e:
        print(f"Error retrieving dataset[0]: {e}")
        return

    sample_bytes = s0.numel() * s0.element_size() + w0.numel() * w0.element_size()
    pinnable = max_pinnable_mb()
    print(f"Sample size ≈ {sample_bytes/1024/1024:.2f} MB, max pinnable ≈ {pinnable} MB")

    if pinnable > 0:
        bs_max = max(1, int((pinnable * 1024 * 1024) // sample_bytes))
        print(f"Estimated max batch size with pin_memory ≈ {bs_max}")
    else:
        bs_max = 0

    print("Starting DataLoader iteration...")
    if args.pin_memory == "true":
        use_pin = True
    elif args.pin_memory == "false":
        use_pin = False
    else:
        use_pin = pinnable >= (sample_bytes * batch_size) / (1024 * 1024)
        if not use_pin:
            print("Pinned memory disabled automatically (batch exceeds pinnable threshold).")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=use_pin,
        shuffle=False,
        collate_fn=collate_stft,
    )

    i = -1
    try:
        for i, _ in enumerate(tqdm(loader, desc="Iterating DataLoader")):
            pass
        print("\n[SUCCESS] Finished iterating the dataset without errors.")
    except Exception as e:
        where = i if i >= 0 else "before first batch"
        print(f"\n[ERROR] DataLoader failed at batch ~{where}")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {e}")
        print("Hints: try pin_memory=False, lower batch size, reduce prefetch/workers, "
              "or increase RLIMIT_MEMLOCK (ulimit -l).")


if __name__ == "__main__":
    main()



