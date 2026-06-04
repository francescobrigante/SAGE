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

# External model storage on Leonardo $FAST (persistent project scratch, 1 TB).
# Foundation models (MERT teacher, etc.) are pre-downloaded here on the login node
# and loaded offline (local_files_only) on the isolated compute nodes.
FAST_DIR = Path(os.getenv("FAST", PROJECT_ROOT / "_fast"))
MODELS_DIR = FAST_DIR / "models"
MERT_MODEL_ID = "m-a-p/MERT-v1-95M"          # HF repo id of the MERT teacher
MERT_MODEL_DIR = MODELS_DIR / "MERT-v1-95M"  # local snapshot dir on $FAST

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
DEFAULT_SILENCE_THRESHOLD = -62
DEFAULT_WANDB_PROJECT = "C-VAE"