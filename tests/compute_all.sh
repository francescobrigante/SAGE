#!/usr/bin/env bash
set -euo pipefail

# Default reference dataset directory; adjust once instead of passing --target-dir each time.
DEFAULT_TARGET_DIR=""
# Default virtualenvs. Set to absolute paths or leave empty to rely on auto-detection / CLI flags.
DEFAULT_MAIN_ENV=""
DEFAULT_METRICS_ENV=""

usage() {
	local main_env_default metrics_env_default
	if [[ -n "$DEFAULT_MAIN_ENV" ]]; then
		main_env_default="$DEFAULT_MAIN_ENV"
	else
		main_env_default="auto (project .venv if present or CLI flag)"
	fi

	if [[ -n "$DEFAULT_METRICS_ENV" ]]; then
		metrics_env_default="$DEFAULT_METRICS_ENV"
	else
		metrics_env_default="(none, pass via --metrics-env)"
	fi

cat <<EOF
Usage: compute_all.sh --output-dir PATH [options]

Required arguments:
	--output-dir PATH     Directory where reconstructed audio will be written.

Optional arguments:
	--checkpoint PATH     Checkpoint file (default: DEFAULT_MODEL_CHECKPOINT from config.py).
	--target-dir PATH     Directory containing the reference audio files (default: DATA_PATH from config.py).
	--main-env VENV       Virtualenv for inference and spectral metrics (default: $main_env_default).
	--metrics-env VENV    Virtualenv for CDPAM/FAD metrics (default: $metrics_env_default).
	--extensions LIST     Comma-separated extensions (e.g. wav,mp3) to filter files.
	--infer-device DEV    Device string for waveform reconstruction (default: DEFAULT_DEVICE from config.py).
	--cdpam-device DEV    Device string for CDPAM (default: DEFAULT_DEVICE from config.py).
	--cdpam-chunk INT     Chunk size in samples for CDPAM (default: 0 -> full clip).
	--csv-dir PATH        Directory where metric CSV/log files will be stored (default: <output-dir>/metrics).
	--skip-cdpam          Skip CDPAM computation.
	--skip-fad            Skip FAD computation.
	--max-files INT       Max number of files to process during inference (default: DEFAULT_MAX_FILES from config.py).
	--help                Show this message.
EOF
}

log() {
	printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

# Colour helpers — mirror ar_spectra.utils.console style
ok()   { printf '\033[1;32m%s\033[0m\n' "$*"; }
info() { printf '\033[0;36m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\033[1;33m%s\033[0m\n' "$*" >&2; }
err()  { printf '\033[1;31m%s\033[0m\n' "$*" >&2; }

# Clean exit on Ctrl+C: kill the whole process group so no Python subprocess keeps running
trap 'warn "Interrupted."; kill 0 2>/dev/null; exit 130' INT TERM

resolve_path() {
	local target="$1"
	if [[ -z "$target" ]]; then
		return 1
	fi
	# 'realpath -m' is Linux-only; use Python as a portable fallback
	if realpath -m "$target" 2>/dev/null; then
		return 0
	fi
	python3 -c "import os, sys; print(os.path.abspath(sys.argv[1]))" "$target"
}

ensure_env() {
	local env_dir="$1"
	local label="$2"
	if [[ -z "$env_dir" ]]; then
		return 0
	fi
	if [[ ! -f "$env_dir/bin/activate" ]]; then
		err "Error: $label virtualenv missing activate script at $env_dir/bin/activate"
		exit 1
	fi
}

run_in_env() {
	local env_dir="$1"
	shift
	if [[ $# -eq 0 ]]; then
		return 0
	fi
	if [[ -n "$env_dir" ]]; then
		(
			set -euo pipefail
			source "$env_dir/bin/activate"
			"$@"
		)
	else
		"$@"
	fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Pull defaults from config.py (single source of truth)
_cfg() { cd "$PROJECT_ROOT" && python -c "from config import $1; print($1)" 2>/dev/null || echo ""; }

if [[ -z "$DEFAULT_TARGET_DIR" ]]; then
	DEFAULT_TARGET_DIR=$(_cfg DATA_PATH)
fi
DEFAULT_DEVICE=$(_cfg DEFAULT_DEVICE)
DEFAULT_MAX_FILES=$(_cfg DEFAULT_MAX_FILES)
DEFAULT_CHECKPOINT=$(_cfg DEFAULT_MODEL_CHECKPOINT)

if [[ -z "$DEFAULT_MAIN_ENV" && -d "$PROJECT_ROOT/.venv" ]]; then
	DEFAULT_MAIN_ENV="$PROJECT_ROOT/.venv"
fi

if [[ -z "$DEFAULT_METRICS_ENV" && -d "$PROJECT_ROOT/.venv_metrics" ]]; then
	DEFAULT_METRICS_ENV="$PROJECT_ROOT/.venv_metrics"
fi

TARGET_DIR="$DEFAULT_TARGET_DIR"
CHECKPOINT="$DEFAULT_CHECKPOINT"
MAIN_ENV="$DEFAULT_MAIN_ENV"
METRICS_ENV="$DEFAULT_METRICS_ENV"
OUTPUT_DIR=""
EXTENSIONS_RAW=""
CDPAM_DEVICE="${DEFAULT_DEVICE:-cpu}"
INFER_DEVICE="${DEFAULT_DEVICE:-cpu}"
CDPAM_CHUNK=0
CSV_DIR=""
SKIP_CDPAM=false
SKIP_FAD=false
MAX_FILES="${DEFAULT_MAX_FILES:-0}"

while [[ $# -gt 0 ]]; do
	case "$1" in
		--target-dir)
			TARGET_DIR="$2"
			shift 2
			;;
		--checkpoint)
			CHECKPOINT="$2"
			shift 2
			;;
		--main-env)
			MAIN_ENV="$2"
			shift 2
			;;
		--metrics-env)
			METRICS_ENV="$2"
			shift 2
			;;
		--output-dir)
			OUTPUT_DIR="$2"
			shift 2
			;;
		--extensions)
			EXTENSIONS_RAW="$2"
			shift 2
			;;
		--cdpam-device)
			CDPAM_DEVICE="$2"
			shift 2
			;;
		--cdpam-chunk)
			CDPAM_CHUNK="$2"
			shift 2
			;;
		--csv-dir)
			CSV_DIR="$2"
			shift 2
			;;
		--infer-device)
			INFER_DEVICE="$2"
			shift 2
			;;
		--skip-cdpam)
			SKIP_CDPAM=true
			shift
			;;
		--skip-fad)
			SKIP_FAD=true
			shift
			;;
		--max-files)
			MAX_FILES="$2"
			shift 2
			;;
		--help|-h)
			usage
			exit 0
			;;
		*)
			err "Unknown argument: $1"
			usage
			exit 1
			;;
	esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
	err "Error: --output-dir is required."
	usage
	exit 1
fi

if [[ -z "$CHECKPOINT" ]]; then
	err "Error: --checkpoint is required (no DEFAULT_MODEL_CHECKPOINT set in config.py)."
	usage
	exit 1
fi

TARGET_DIR="$(resolve_path "$TARGET_DIR")"
CHECKPOINT="$(resolve_path "$CHECKPOINT")"

if [[ -n "$MAIN_ENV" ]]; then
	MAIN_ENV="$(resolve_path "$MAIN_ENV")"
fi

if [[ -n "$METRICS_ENV" ]]; then
	METRICS_ENV="$(resolve_path "$METRICS_ENV")"
fi

OUTPUT_DIR="$(resolve_path "$OUTPUT_DIR")"

if [[ -z "$CSV_DIR" ]]; then
	CSV_DIR="$OUTPUT_DIR/metrics"
fi

CSV_DIR="$(resolve_path "$CSV_DIR")"

if [[ -z "$METRICS_ENV" ]] && ( ! $SKIP_CDPAM || ! $SKIP_FAD ); then
	warn "Warning: no metrics virtualenv specified. CDPAM and FAD will run in the current environment."
fi

if [[ ! -d "$TARGET_DIR" ]]; then
	err "Error: target directory not found -> $TARGET_DIR"
	exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
	err "Error: checkpoint file not found -> $CHECKPOINT"
	exit 1
fi

mkdir -p "$OUTPUT_DIR" "$CSV_DIR"

ensure_env "$MAIN_ENV" "main"
ensure_env "$METRICS_ENV" "metrics"

IFS=',' read -r -a EXT_ARRAY <<< "$EXTENSIONS_RAW"
declare -a EXT_LIST=()
for ext in "${EXT_ARRAY[@]+"${EXT_ARRAY[@]}"}"; do
	ext="${ext//[[:space:]]/}"
	if [[ -z "$ext" ]]; then
		continue
	fi
	if [[ ${ext:0:1} != '.' ]]; then
		ext=".$ext"
	fi
	EXT_LIST+=("$ext")
done

CLI_EXT=""
if [[ ${#EXT_LIST[@]} -gt 0 ]]; then
	for ext in "${EXT_LIST[@]}"; do
		ext_no_dot="${ext#.}"
		if [[ -n "$CLI_EXT" ]]; then
			CLI_EXT+=",$ext_no_dot"
		else
			CLI_EXT="$ext_no_dot"
		fi
	done
fi

info "Target dir: $TARGET_DIR"
info "Checkpoint: $CHECKPOINT"
info "Output dir: $OUTPUT_DIR"
info "Metrics dir: $CSV_DIR"
info "Inference device: $INFER_DEVICE"

GEN_EXT=""
if [[ ${#EXT_LIST[@]} -gt 0 ]]; then
	GEN_EXT="$(printf "%s," "${EXT_LIST[@]}")"
	GEN_EXT="${GEN_EXT%,}"
fi

info "Running inference via evaluate.py"
declare -a GEN_CMD=(python "$PROJECT_ROOT/evaluate.py" \
	--model-checkpoint "$CHECKPOINT" \
	--target-dir "$TARGET_DIR" \
	--output-dir "$OUTPUT_DIR" \
	--device "$INFER_DEVICE")
if [[ -n "$GEN_EXT" ]]; then
	GEN_CMD+=(--extensions "$GEN_EXT")
fi
if [[ "$MAX_FILES" -gt 0 ]]; then
	GEN_CMD+=(--max-files "$MAX_FILES")
fi
run_in_env "$MAIN_ENV" "${GEN_CMD[@]}"

if [[ ! -d "$OUTPUT_DIR" ]]; then
	err "Error: expected predictions in $OUTPUT_DIR but directory not found."
	exit 1
fi

declare -a SPECTRAL_CMD=(python "$PROJECT_ROOT/tests/compute_spectral.py" --target-dir "$TARGET_DIR" --preds-dir "$OUTPUT_DIR")
if [[ -n "$CLI_EXT" ]]; then
	SPECTRAL_CMD+=(--extensions "$CLI_EXT")
fi
SPECTRAL_CMD+=(--csv_out "$CSV_DIR/spectral.csv")

info "Computing spectral metrics"
run_in_env "$MAIN_ENV" "${SPECTRAL_CMD[@]}"

if ! $SKIP_CDPAM; then
	declare -a CDPAM_CMD=(python "$PROJECT_ROOT/tests/compute_cdpam.py" --target-dir "$TARGET_DIR" --preds-dir "$OUTPUT_DIR" --device "$CDPAM_DEVICE")
	if [[ -n "$CLI_EXT" ]]; then
		CDPAM_CMD+=(--extensions "$CLI_EXT")
	fi
	if [[ "$CDPAM_CHUNK" -gt 0 ]]; then
		CDPAM_CMD+=(--chunk_size "$CDPAM_CHUNK")
	fi
	CDPAM_CMD+=(--csv_out "$CSV_DIR/cdpam.csv")
	info "Computing CDPAM"
	run_in_env "${METRICS_ENV:-$MAIN_ENV}" "${CDPAM_CMD[@]}"
fi

if ! $SKIP_FAD; then
	declare -a FAD_CMD=(python "$PROJECT_ROOT/tests/compute_fad.py" --target-dir "$TARGET_DIR" --preds-dir "$OUTPUT_DIR")
	if [[ "$MAX_FILES" -gt 0 ]]; then
		FAD_CMD+=(--max-files "$MAX_FILES")
	fi
	info "Computing FAD"
	run_in_env "${METRICS_ENV:-$MAIN_ENV}" "${FAD_CMD[@]}" | tee "$CSV_DIR/fad.txt"
fi

ok "All requested computations completed."
