import argparse
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Add project root to sys.path to safely import config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try loading default values from config just like the bash script
try:
    from config import DATA_PATH, DEFAULT_DEVICE, DEFAULT_MAX_FILES, DEFAULT_MODEL_CHECKPOINT
except ImportError:
    DATA_PATH = ""
    DEFAULT_DEVICE = "cpu"
    DEFAULT_MAX_FILES = 0
    DEFAULT_MODEL_CHECKPOINT = ""

def log_info(msg):
    """Print info logs in cyan (mirrors console.py style)."""
    print(f"\033[0;36m[{datetime.now().strftime('%H:%M:%S')}] {msg}\033[0m")

def log_ok(msg):
    """Print success logs in green."""
    print(f"\033[1;32m{msg}\033[0m")

def log_err(msg):
    """Print error logs in red."""
    print(f"\033[1;31m{msg}\033[0m", file=sys.stderr)

def get_python_exe(env_path=None):
    """Returns the correct Python executable depending on OS and virtualenv."""
    if not env_path:
        # Defaults to the Python executable currently running the script (e.g. from 'uv run')
        return sys.executable
    
    env_dir = Path(env_path)
    # Handle Windows vs Unix
    if os.name == 'nt':  # Windows
        python_exe = env_dir / "Scripts" / "python.exe"
    else:                # Unix (Mac/Linux)
        python_exe = env_dir / "bin" / "python"
        
    if not python_exe.exists():
        log_err(f"Error: Python executable not found at {python_exe}")
        sys.exit(1)
        
    return str(python_exe)

def run_command(cmd, env_path=None, critical=True):
    """Executes a subprocess command safely.

    Args:
        critical: if True (default), exit the process on failure; if False, log
                  the error and return the exit code so the pipeline can continue.
    """
    cmd[0] = get_python_exe(env_path)
    try:
        subprocess.run(cmd, check=True)
        return 0
    except subprocess.CalledProcessError as e:
        log_err(f"Command failed with exit code {e.returncode}")
        if critical:
            sys.exit(e.returncode)
        return e.returncode
    except KeyboardInterrupt:
        log_err("\nInterrupted by user. Exiting...")
        sys.exit(130)

def main():
    parser = argparse.ArgumentParser(description="Cross-platform computation script")
    
    # Required/Output
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for reconstructed audio.")
    
    # Optional configs
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL_CHECKPOINT, help="Checkpoint file path.")
    parser.add_argument("--target-dir", type=Path, default=DATA_PATH, help="Directory containing reference audio files.")
    parser.add_argument("--csv-dir", type=Path, help="Directory where metric CSV/log files will be stored.")
    parser.add_argument("--extensions", type=str, help="Comma-separated extensions (e.g. wav,mp3).")
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES, help="Max files to process (inference).")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for all evaluation steps.")
    parser.add_argument("--cache-dir", type=Path, help="Shared directory for target embeddings cache.")
    
    # Virtualenvs
    parser.add_argument("--main-env", type=Path, help="Virtualenv for inference and spectral metrics.")
    parser.add_argument("--metrics-env", type=Path, help="Virtualenv for CDPAM/FAD metrics.")
    
    # Devices & specific configs
    parser.add_argument("--infer-device", type=str, default=DEFAULT_DEVICE, help="Device for waveform reconstruction.")
    parser.add_argument("--cdpam-device", type=str, default=DEFAULT_DEVICE, help="Device for CDPAM.")
    parser.add_argument("--cdpam-chunk", type=int, default=0, help="Chunk size in samples for CDPAM.")
    
    # Skipping steps
    parser.add_argument("--skip-cdpam", action="store_true", help="Skip CDPAM computation.")
    parser.add_argument("--skip-fad", action="store_true", help="Skip FAD (fadtk) computation.")
    parser.add_argument("--skip-clap", action="store_true", help="Skip CLAP-LAION cosine score.")
    parser.add_argument("--skip-fad-gudgud", action="store_true", help="Skip FAD (gudgud96) computation.")

    # Model selection for new steps
    parser.add_argument("--clap-model", default="both", choices=["music", "audio", "both"],
                        help="CLAP flavour(s) to compute (default: both).")
    parser.add_argument("--fad-model", default="mert",
                        choices=["vggish", "clap-laion", "clap-laion-audio", "mert"],
                        help="Backbone for FAD fadtk (default: mert).")
    parser.add_argument("--fad-gudgud-model", default="clap-audio",
                        choices=["vggish", "pann", "clap-music", "clap-audio", "encodec"],
                        help="Backbone for FAD gudgud96 (default: clap-audio).")

    args = parser.parse_args()

    # Validations
    if not args.checkpoint:
        parser.error("--checkpoint is required (no DEFAULT_MODEL_CHECKPOINT set).")
    if not args.target_dir or not args.target_dir.is_dir():
        log_err(f"Error: target directory not found -> {args.target_dir}")
        sys.exit(1)
    if not args.checkpoint.is_file():
        log_err(f"Error: checkpoint file not found -> {args.checkpoint}")
        sys.exit(1)

    # Directories setup
    csv_dir = args.csv_dir if args.csv_dir else args.output_dir / "metrics"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Extensions parsing
    cli_ext, gen_ext = "", ""
    if args.extensions:
        exts = [e.strip() for e in args.extensions.split(',') if e.strip()]
        cli_ext = ",".join([e.lstrip('.') for e in exts])
        gen_ext = ",".join([e if e.startswith('.') else f".{e}" for e in exts])

    # Initialization summary
    log_info(f"Target dir: {args.target_dir}")
    log_info(f"Checkpoint: {args.checkpoint}")
    log_info(f"Output dir: {args.output_dir}")
    log_info(f"Metrics dir: {csv_dir}")
    log_info(f"Inference device: {args.infer_device}")

    # ==========================
    # 1. EVALUATE / INFERENCE
    # ==========================
    log_info("Running inference via evaluate.py")
    gen_cmd = ["python", str(PROJECT_ROOT / "evaluate.py"), 
               "--model-checkpoint", str(args.checkpoint),
               "--target-dir", str(args.target_dir), 
               "--output-dir", str(args.output_dir), 
               "--device", args.infer_device]
    if gen_ext: gen_cmd.extend(["--extensions", gen_ext])
    if args.max_files > 0: gen_cmd.extend(["--max-files", str(args.max_files)])
    
    run_command(gen_cmd, args.main_env)

    # ==========================
    # 2. SPECTRAL METRICS
    # ==========================
    log_info("Computing spectral metrics")
    spectral_cmd = ["python", str(PROJECT_ROOT / "evaluation/compute_spectral.py"), 
                    "--target-dir", str(args.target_dir), 
                    "--preds-dir", str(args.output_dir),
                    "--csv_out", str(csv_dir / "spectral.csv"),
                    "--batch-size", str(args.batch_size)]
    if cli_ext: spectral_cmd.extend(["--extensions", cli_ext])
    if args.max_files > 0: spectral_cmd.extend(["--max-files", str(args.max_files)])
    
    run_command(spectral_cmd, args.main_env)

    # ==========================
    # 3. CLAP SCORE
    # ==========================
    if not args.skip_clap:
        log_info("Computing CLAP-LAION cosine score")
        clap_cmd = ["python", str(PROJECT_ROOT / "evaluation/compute_clap_score.py"),
                    "--target-dir", str(args.target_dir),
                    "--preds-dir", str(args.output_dir),
                    "--model", args.clap_model,
                    "--csv_out", str(csv_dir / "clap_score.csv"),
                    "--batch-size", str(args.batch_size)]
        if args.cache_dir: clap_cmd.extend(["--cache-dir", str(args.cache_dir)])
        if args.max_files > 0: clap_cmd.extend(["--max-files", str(args.max_files)])
        run_command(clap_cmd, args.metrics_env or args.main_env)

    # ==========================
    # 4. CDPAM
    # ==========================
    if not args.skip_cdpam:
        log_info("Computing CDPAM")
        cdpam_cmd = ["python", str(PROJECT_ROOT / "evaluation/compute_cdpam.py"), 
                     "--target-dir", str(args.target_dir), 
                     "--preds-dir", str(args.output_dir), 
                     "--device", args.cdpam_device,
                     "--csv_out", str(csv_dir / "cdpam.csv"),
                     "--batch-size", str(args.batch_size)]
        if cli_ext: cdpam_cmd.extend(["--extensions", cli_ext])
        if args.cdpam_chunk > 0: cdpam_cmd.extend(["--chunk_size", str(args.cdpam_chunk)])
        if args.max_files > 0: cdpam_cmd.extend(["--max-files", str(args.max_files)])
        
        run_command(cdpam_cmd, args.metrics_env or args.main_env)

    # ==========================
    # 5. FAD GUDGUD (clap-audio)
    # ==========================
    if not args.skip_fad_gudgud:
        log_info("Computing FAD (gudgud96 / clap-audio)")
        fad_gudgud_cmd = ["python", str(PROJECT_ROOT / "evaluation/compute_fad_gudgud.py"),
                          "--target-dir", str(args.target_dir),
                          "--preds-dir", str(args.output_dir),
                          "--model", args.fad_gudgud_model,
                          "--csv_out", str(csv_dir / "fad_gudgud.csv"),
                          "--batch-size", str(args.batch_size)]
        if args.cache_dir: fad_gudgud_cmd.extend(["--cache-dir", str(args.cache_dir)])
        if args.max_files > 0: fad_gudgud_cmd.extend(["--max-files", str(args.max_files)])
        run_command(fad_gudgud_cmd, args.metrics_env or args.main_env, critical=False)

    # ==========================
    # 6. FAD FADTK (mert)
    # ==========================
    if not args.skip_fad:
        log_info("Computing FAD (fadtk / mert)")
        fad_cmd = ["python", str(PROJECT_ROOT / "evaluation/compute_fad.py"),
                   "--target-dir", str(args.target_dir),
                   "--preds-dir", str(args.output_dir),
                   "--model", args.fad_model,
                   "--csv_out", str(csv_dir / "fad_mert.csv"),
                   "--batch-size", str(args.batch_size)]
        if args.cache_dir: fad_cmd.extend(["--cache-dir", str(args.cache_dir)])
        if args.max_files > 0: fad_cmd.extend(["--max-files", str(args.max_files)])
        run_command(fad_cmd, args.metrics_env or args.main_env, critical=False)


if __name__ == "__main__":
    main()
