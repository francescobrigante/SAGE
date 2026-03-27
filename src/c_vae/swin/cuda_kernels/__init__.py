# ===============================================================
# Fused CUDA kernels for Swin Transformer window operations.
# Provides WindowProcess (roll + partition) and WindowProcessReverse
# (merge + roll) as single-pass GPU ops. Falls back gracefully to
# None when the compiled extension is not available (MPS, CPU).
# Build: cd src/c_vae/swin/cuda_kernels && python setup.py install
# ===============================================================

try:
    from .window_process import WindowProcess, WindowProcessReverse
    FUSED_WINDOW_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    WindowProcess = None
    WindowProcessReverse = None
    FUSED_WINDOW_AVAILABLE = False
