# ===================================================================
# Centralized configuration
#
#   contains global configuration constants and paths
# ===================================================================

from pathlib import Path
import torch

# Project Roots
PROJECT_ROOT = Path(__file__).parent.resolve()
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RUNS_DIR = PROJECT_ROOT / "runs"
CONFIG_DIR = PROJECT_ROOT / "config"

# Default configuration constants
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_DEVICE = "mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu"

# Add other project-wide constants here to use as single-source-of-truth.
# FMA_METADATA = "/Users/francesco/Desktop/fma_metadata/tracks.csv"
FMA_METADATA = "C:/users/franc/Desktop/fma_metadata/tracks.csv"
# DATA_PATH = "/Users/francesco/Desktop/fma_small"
DATA_PATH = "C:/users/franc/Desktop/fma_small"

DEFAULT_BATCH_SIZE = 2
DEFAULT_MAX_FILES = 0 # 0 means all

DEFAULT_MODEL_CHECKPOINT = CHECKPOINT_DIR / "eulerodec.ckpt"
