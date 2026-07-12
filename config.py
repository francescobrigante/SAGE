# ===================================================================
# Centralized configuration
#
#   contains global configuration constants and paths
# ===================================================================

from pathlib import Path
import torch
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project Roots
PROJECT_ROOT = Path(__file__).parent.resolve()
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RUNS_DIR = PROJECT_ROOT / "runs"
CONFIG_DIR = PROJECT_ROOT / "config"

# Default configuration constants
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_DEVICE = "mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT_AUDIO_CHANNELS = 2

# Add other project-wide constants here to use as single-source-of-truth.
FMA_METADATA = os.getenv("FMA_METADATA")
DATA_PATH = os.getenv("DATA_PATH")

# Multi-corpus training corpora (future IscrC_MM run): FMA-full + MTG-Jamendo + M4Singer.
# Resolved in Hydra via ${config:NAME}. Paths come from .env (cross-account scratch);
# unset on machines without the corpora, so the single-corpus FMA path stays unaffected.
FMA_FULL_AUDIO = os.getenv("FMA_FULL_AUDIO")              # FMA-full mp3 root (106k full-length tracks)
JAMENDO_AUDIO = os.getenv("JAMENDO_AUDIO")                # MTG-Jamendo mp3 root (XX/ID.mp3 layout)
JAMENDO_SPLIT_TSV = os.getenv("JAMENDO_SPLIT_TSV")        # split-0 autotagging-train.tsv (leakage-free train)
M4SINGER_AUDIO = os.getenv("M4SINGER_AUDIO")             # M4Singer wav root (scanned recursively)

# External model storage on Leonardo $FAST (persistent project scratch, 1 TB).
# Foundation models (CLAP teacher, etc.) are pre-downloaded here on the login node
# and loaded offline on the isolated compute nodes.
FAST_DIR = Path(os.getenv("FAST", PROJECT_ROOT / "_fast"))
MODELS_DIR = FAST_DIR / "models"

# MoisesDB — zero-shot reconstruction-validation set. ``MOISESDB_MIX_ORIGINAL``
# is the exact 10 s mixtures directory used for the paper's recon validation
# (chunks_mix_original/original, 1998 × 10 s WAVs @ 44.1 kHz stereo). The set was
# imported as-is; its filename UUIDs do not map to moisesdb_v0.1, so no genre.
MOISESDB_MIX_ORIGINAL = Path(os.getenv(
    "MOISESDB_MIX_ORIGINAL",
    str(FAST_DIR / "datasets" / "moisesdb" / "chunks_mix_original" / "original")))

# Filelist cache for the multi-corpus build: the provider-scan + per-file probe-filter
# over ~138k files costs ~11 min at every job start (and every --requeue resume). The
# filtered, sorted list is deterministic and the files are immutable, so we persist it
# here as a .txt and reload it on subsequent runs (build → ~1 s). Opt-in per corpus via
# `filelist_cache:` in multicorpus.yaml; single-corpus FMA never sets it. Delete the .txt
# to force a rebuild after changing a corpus.
FILELIST_CACHE_DIR = FAST_DIR / "filelist_cache"

DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_FILES = 0 # 0 means all

DEFAULT_MODEL_CHECKPOINT = CHECKPOINT_DIR / "eulerodec.ckpt"

# Training & Dataloading Defaults
DEFAULT_NUM_WORKERS = 8
DEFAULT_DATALOADER_TIMEOUT = 120
DEFAULT_AUDIO_LOAD_TIMEOUT = 60   # per-file timeout (s) to abort stuck MP3 decode in worker
DEFAULT_SEED = 94
DEFAULT_MAX_RETRIES_PER_SAMPLE = 8
DEFAULT_MAX_PAD_RATIO = 0.05
DEFAULT_AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus")

# Partial-read: decode only a ~segment-long window (via torchaudio frame_offset/num_frames)
# instead of the whole file, then crop. Big win on multi-minute tracks (FMA-full, Jamendo).
# Off by default for full backward compatibility; enable per-dataset in config for long-file corpora.
DEFAULT_PARTIAL_READ = False
# Source-domain samples of slack loaded around the window so that, after resampling
# (sinc edge transients) and SR rounding, we always retain >= segment_samples to crop cleanly.
DEFAULT_PARTIAL_READ_MARGIN = 4096
DEFAULT_SILENCE_THRESHOLD = -62
DEFAULT_WANDB_PROJECT = "C-VAE"