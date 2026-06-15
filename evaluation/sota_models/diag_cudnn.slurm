#!/bin/bash
# ============================================================
# diag_cudnn.slurm — pinpoint the CUDNN_STATUS_NOT_INITIALIZED in the sota venv.
# Tests a cuDNN conv (a) with cineca-ai LD_LIBRARY_PATH, (b) with it unset.
# ============================================================
#SBATCH --job-name=diag_cudnn
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=logs/diag_cudnn_%j.out
#SBATCH --error=logs/diag_cudnn_%j.err

set -uo pipefail
mkdir -p logs

module load profile/deeplrn cineca-ai/4.3.0
unset PYTHONPATH
source $WORK/.envs/sota/bin/activate

echo "=== nvidia-smi ==="; nvidia-smi | head -15

CONV='
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| cudnn", torch.backends.cudnn.version())
print("LD_LIBRARY_PATH=", __import__("os").environ.get("LD_LIBRARY_PATH","")[:300])
x = torch.randn(1,4,64,64, device="cuda")
w = torch.randn(8,4,3,3, device="cuda")
try:
    y = torch.nn.functional.conv2d(x, w, padding=1); torch.cuda.synchronize()
    print("CONV2D (cuDNN) OK ->", tuple(y.shape))
except Exception as e:
    print("CONV2D FAILED:", type(e).__name__, str(e)[:160])
try:
    z = (x.flatten(1) @ x.flatten(1).T); torch.cuda.synchronize()
    print("MATMUL (no cuDNN) OK ->", tuple(z.shape))
except Exception as e:
    print("MATMUL FAILED:", type(e).__name__, str(e)[:160])
'

echo ""; echo "=== (A) WITH cineca LD_LIBRARY_PATH ==="
python -u -c "$CONV"

echo ""; echo "=== (B) WITH LD_LIBRARY_PATH UNSET (use torch-bundled cu126 libs) ==="
env -u LD_LIBRARY_PATH python -u -c "$CONV"

echo ""; echo "=== where does libcudnn load from in each case? ==="
echo "--- bundled cudnn in venv ---"; ls $WORK/.envs/sota/lib/python3.11/site-packages/nvidia/cudnn/lib/ 2>/dev/null | head
