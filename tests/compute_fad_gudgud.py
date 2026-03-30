# ===============================================================================
# compute_fad_gudgud.py
# FAD via frechet_audio_distance (gudgud96). Independent FAD implementation with
# VGGish, PANN, CLAP (music/audio), and EnCodec backbones. Auto-resamples input.
# ===============================================================================
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_PATH, RUNS_DIR
from ar_spectra.utils.console import ok, warn, err, info

from frechet_audio_distance import FrechetAudioDistance

# Backbone registry: maps CLI --model values to FrechetAudioDistance init kwargs.
# sample_rate is the model's native SR; gudgud96 resamples source audio automatically.
_BACKBONE_CONFIGS: dict[str, dict] = {
    "vggish":     {"model_name": "vggish",  "sample_rate": 16000},
    "pann":       {"model_name": "pann",    "sample_rate": 16000},
    # submodel_name must match the short names validated inside FrechetAudioDistance.__init__,
    # not the checkpoint filenames. "music_audioset" → music_audioset_epoch_15_esc_90.14.pt
    "clap-music": {"model_name": "clap",    "sample_rate": 48000,
                   "submodel_name": "music_audioset"},
    # "630k-audioset" + enable_fusion=True → 630k-audioset-fusion-best.pt
    "clap-audio": {"model_name": "clap",    "sample_rate": 48000,
                   "submodel_name": "630k-audioset", "enable_fusion": True},
    "encodec":    {"model_name": "encodec", "sample_rate": 24000},
}


def main():
    parser = argparse.ArgumentParser(
        description="FAD via frechet_audio_distance (gudgud96)."
    )
    parser.add_argument("--target-dir", default=str(DATA_PATH))
    parser.add_argument("--preds-dir", default=str(RUNS_DIR / "inference"))
    parser.add_argument(
        "--model", default="vggish", choices=list(_BACKBONE_CONFIGS.keys()),
        help="Backbone model for FAD (default: vggish)"
    )
    args = parser.parse_args()

    cfg = _BACKBONE_CONFIGS[args.model]

    info(f"Target directory:      {args.target_dir}")
    info(f"Predictions directory: {args.preds_dir}")
    info(f"Backbone:              {args.model} ({cfg['model_name']}, {cfg['sample_rate']} Hz)")

    warn("FAD is a distributional metric. Results are only statistically meaningful with a large number of samples.")

    try:
        frechet = FrechetAudioDistance(**cfg)
        fad_score = frechet.score(args.target_dir, args.preds_dir)
        ok(f"FAD gudgud ({args.model}): {fad_score}")
    except KeyboardInterrupt:
        warn("[Interrupted] FAD computation aborted.")
    except Exception as e:
        err(f"FAD computation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
