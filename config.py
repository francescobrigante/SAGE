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

DEFAULT_BATCH_SIZE = 2
DEFAULT_MAX_FILES = 0 # 0 means all

DEFAULT_MODEL_CHECKPOINT = CHECKPOINT_DIR / "eulerodec.ckpt"

# Training & Dataloading Defaults
DEFAULT_NUM_WORKERS = 8
DEFAULT_DATALOADER_TIMEOUT = 60
DEFAULT_WANDB_PROJECT = "C-VAE"
DEFAULT_SEED = 94
DEFAULT_MAX_RETRIES_PER_SAMPLE = 8
DEFAULT_MAX_PAD_RATIO = 0.05
DEFAULT_AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus")
DEFAULT_SILENCE_THRESHOLD = -62
