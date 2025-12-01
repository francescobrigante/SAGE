#!/usr/bin/env bash
set -euo pipefail

# Default reference dataset directory; adjust once instead of passing --target-dir each time.
DEFAULT_TARGET_DIR="/home/cerovaz/repos/data/jamendo_full/test_trimmed"
# Default virtualenvs. Set to absolute paths or leave empty to rely on auto-detection / CLI flags.
DEFAULT_MAIN_ENV="/home/cerovaz/repos/ICML/Eulero_BackBone/.venv"
DEFAULT_METRICS_ENV="/home/cerovaz/repos/ICML/Eulero_BackBone/.venv_test"

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
Usage: compute_all.sh --checkpoint PATH --output-dir PATH [options]

Required arguments:
	--checkpoint PATH     Checkpoint file to use for inference.
	--output-dir PATH     Directory where reconstructed audio will be written.

Optional arguments:
	--target-dir PATH     Directory containing the reference audio files (default: $DEFAULT_TARGET_DIR).
	--main-env VENV       Virtualenv for inference and spectral metrics (default: $main_env_default).
	--metrics-env VENV    Virtualenv for CDPAM/FAD metrics (default: $metrics_env_default).
	--extensions LIST     Comma-separated extensions (e.g. wav,mp3) to filter files.
	--infer-device DEV    Device string for waveform reconstruction (default: cuda:0).
	--cdpam-device DEV    Device string for CDPAM (default: cuda:0).
	--cdpam-chunk INT     Chunk size in samples for CDPAM (default: 0 -> full clip).
	--csv-dir PATH        Directory where metric CSV/log files will be stored (default: <output-dir>/metrics).
	--skip-cdpam          Skip CDPAM computation.
	--skip-fad            Skip FAD computation.
	--help                Show this message.
EOF
}

log() {
	printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

resolve_path() {
	local target="$1"
	if [[ -z "$target" ]]; then
		return 1
	fi
	realpath -m "$target"
}

ensure_env() {
	local env_dir="$1"
	local label="$2"
	if [[ -z "$env_dir" ]]; then
		return 0
	fi
	if [[ ! -f "$env_dir/bin/activate" ]]; then
		echo "Error: $label virtualenv missing activate script at $env_dir/bin/activate" >&2
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

if [[ -z "$DEFAULT_MAIN_ENV" && -d "$PROJECT_ROOT/.venv" ]]; then
	DEFAULT_MAIN_ENV="$PROJECT_ROOT/.venv"
fi

if [[ -z "$DEFAULT_METRICS_ENV" && -d "$PROJECT_ROOT/.venv_metrics" ]]; then
	DEFAULT_METRICS_ENV="$PROJECT_ROOT/.venv_metrics"
fi

TARGET_DIR="$DEFAULT_TARGET_DIR"
CHECKPOINT=""
MAIN_ENV="$DEFAULT_MAIN_ENV"
METRICS_ENV="$DEFAULT_METRICS_ENV"
OUTPUT_DIR=""
EXTENSIONS_RAW=""
CDPAM_DEVICE="cuda:0"
INFER_DEVICE="cuda:0"
CDPAM_CHUNK=0
CSV_DIR=""
SKIP_CDPAM=false
SKIP_FAD=false

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
		--help|-h)
			usage
			exit 0
			;;
		*)
			echo "Unknown argument: $1" >&2
			usage
			exit 1
			;;
	esac
done

if [[ -z "$CHECKPOINT" || -z "$OUTPUT_DIR" ]]; then
	echo "Error: --checkpoint and --output-dir are required." >&2
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

if [[ -z "$METRICS_ENV" ]]; then
	echo "Error: metrics virtualenv not specified. Set DEFAULT_METRICS_ENV or pass --metrics-env." >&2
	usage
	exit 1
fi

if [[ ! -d "$TARGET_DIR" ]]; then
	echo "Error: target directory not found -> $TARGET_DIR" >&2
	exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
	echo "Error: checkpoint file not found -> $CHECKPOINT" >&2
	exit 1
fi

mkdir -p "$OUTPUT_DIR" "$CSV_DIR"

ensure_env "$MAIN_ENV" "main"
ensure_env "$METRICS_ENV" "metrics"

IFS=',' read -r -a EXT_ARRAY <<< "$EXTENSIONS_RAW"
declare -a EXT_LIST=()
for ext in "${EXT_ARRAY[@]}"; do
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

log "Target dir: $TARGET_DIR"
log "Checkpoint: $CHECKPOINT"
log "Output dir: $OUTPUT_DIR"
log "Metrics dir: $CSV_DIR"
log "Inference device: $INFER_DEVICE"

GEN_EXT=""
if [[ ${#EXT_LIST[@]} -gt 0 ]]; then
	GEN_EXT="$(printf "%s," "${EXT_LIST[@]}")"
	GEN_EXT="${GEN_EXT%,}"
fi

log "Running inference via generate_test_preds.py"
declare -a GEN_CMD=(python "$PROJECT_ROOT/test_metrics/generate_test_preds.py" \
	--model-checkpoint "$CHECKPOINT" \
	--target-dir "$TARGET_DIR" \
	--output-dir "$OUTPUT_DIR" \
	--device "$INFER_DEVICE")
if [[ -n "$GEN_EXT" ]]; then
	GEN_CMD+=(--extensions "$GEN_EXT")
fi
run_in_env "$MAIN_ENV" "${GEN_CMD[@]}"

if [[ ! -d "$OUTPUT_DIR" ]]; then
	echo "Error: expected predictions in $OUTPUT_DIR but directory not found." >&2
	exit 1
fi

declare -a SPECTRAL_CMD=(python "$PROJECT_ROOT/test_metrics/compute_spectral.py" --target-dir "$TARGET_DIR" --preds-dir "$OUTPUT_DIR")
if [[ -n "$CLI_EXT" ]]; then
	SPECTRAL_CMD+=(--extensions "$CLI_EXT")
fi
SPECTRAL_CMD+=(--csv_out "$CSV_DIR/spectral.csv")

log "Computing spectral metrics"
run_in_env "$MAIN_ENV" "${SPECTRAL_CMD[@]}"

if ! $SKIP_CDPAM; then
	declare -a CDPAM_CMD=(python "$PROJECT_ROOT/test_metrics/compute_cdpam.py" --target-dir "$TARGET_DIR" --preds-dir "$OUTPUT_DIR" --device "$CDPAM_DEVICE")
	if [[ -n "$CLI_EXT" ]]; then
		CDPAM_CMD+=(--extensions "$CLI_EXT")
	fi
	if [[ "$CDPAM_CHUNK" -gt 0 ]]; then
		CDPAM_CMD+=(--chunk_size "$CDPAM_CHUNK")
	fi
	CDPAM_CMD+=(--csv_out "$CSV_DIR/cdpam.csv")
	log "Computing CDPAM"
	run_in_env "$METRICS_ENV" "${CDPAM_CMD[@]}"
fi

if ! $SKIP_FAD; then
	log "Computing FAD"
	if [[ -n "$METRICS_ENV" ]]; then
		(
			set -euo pipefail
			source "$METRICS_ENV/bin/activate"
			python "$PROJECT_ROOT/test_metrics/compute_fad.py" --target-dir "$TARGET_DIR" --preds-dir "$OUTPUT_DIR"
		) | tee "$CSV_DIR/fad.txt"
	else
		python "$PROJECT_ROOT/test_metrics/compute_fad.py" --target-dir "$TARGET_DIR" --preds-dir "$OUTPUT_DIR" | tee "$CSV_DIR/fad.txt"
	fi
fi

log "All requested computations completed."
