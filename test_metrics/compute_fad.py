## WARNING: THIS IS NOT THE OFFICIAL FAD IMPEMENTATION. BOTH FOR THE OFFICIAL VERSION AND THIS ONE USE ANOTHER ENV

from frechet_audio_distance import FrechetAudioDistance
from argparse import ArgumentParser
from rich.console import Console
console = Console()

def ok(msg: str) -> None:
    console.print(msg, style="bold green")


def warn(msg: str) -> None:
    console.print(msg, style="bold yellow")


def err(msg: str) -> None:
    console.print(msg, style="bold red")
    
parser = ArgumentParser(description="Compute Frechet Audio Distance")
parser.add_argument("--target-dir", type=str, default="/home/cerovaz/repos/data/jamendo_full/test_trimmed", required=False, help="Path to the target/reference audio directory")
parser.add_argument("--preds-dir", type=str, default="/home/cerovaz/repos/ICML/Eulero_BackBone/runs/inference/111_tris_cplx_24epoch", required=False, help="Path to the predicted/generated audio directory")

args = parser.parse_args()

target_dir = args.target_dir if isinstance(args.target_dir, str) else str(args.target_dir)
preds_dir = args.preds_dir if isinstance(args.preds_dir, str) else str(args.preds_dir)

frechet = FrechetAudioDistance(
    model_name="clap",
    sample_rate=48000,
    submodel_name="music_audioset"
)

fad_score = frechet.score(
    target_dir,
    preds_dir,
    dtype="float32"
)

ok("Computed FAD score between")
ok(f"Target directory: {target_dir}")
ok(f"Predictions directory: {preds_dir}")

ok(f"FAD: {fad_score}")