# Paper runs

Every result of the paper and the command that reproduces it. Commands run from the
repository root in the environment of `uv sync --extra train --extra eval` (add
`--extra baselines` for the baseline rows). On a SLURM cluster the same commands go
through the templates in `scripts/slurm/` (`train.sbatch <recipe> <overrides>`,
`eval.sbatch <script> <overrides>`; pass `-A <account> -p <partition>` to `sbatch`).

## Paths

Data, weights and outputs are set in one place, `configs/paths/default.yaml`: every
entry reads an environment variable (`FMA_AUDIO`, `FMA_METADATA`, `SAGE_MODELS`, ...).
Export them, or copy the file to `configs/paths/local.yaml` (git-ignored), fill it in
and add `paths=local` to every command. One value can also be set on the command line,
e.g. `paths.musiccaps=/data/musiccaps_10s`. The weights go under `SAGE_MODELS`
(default `models/`): `SAGE_FTe992.ckpt`, `LAION_CLAP/music_audioset_epoch_15_esc_90.14.pt`
(teacher and MAEB oracle) and `PANN/Cnn14_16k_mAP=0.438.pth` (FAD-PANN).

## Training (Table 6)

See the [training guide](training.md) for data and teacher setup, GPU sizing,
logging, checkpoint resuming, and SLURM execution.

| Result | Command |
|---|---|
| Phase 1, pretraining (500 epochs, 16 GPUs, global batch 128) | `python train.py +experiment=pretrain trainer.trainer.num_gpus=4 ++trainer.trainer.num_nodes=4` |
| Phase 2, decoder fine-tuning → `SAGE_FTe992.ckpt` (epoch 992) | `python train.py +experiment=decoder_ft +init_from=<phase-1 checkpoint> trainer.trainer.num_gpus=4 ++trainer.trainer.num_nodes=4` |

The recipes (`configs/experiment/pretrain.yaml`, `decoder_ft.yaml`) hold every value of
Table 6; `tests/test_recipes.py` checks them. The batch size is global, so fewer GPUs run
the same recipe.

## Reconstruction (Table 2)

Five evaluation sets (`configs/dataset/`), two protocols:

| `dataset=` | Set | Protocol | Path key |
|---|---|---|---|
| `fma` | FMA test (11,263 clips of 30 s) | `fma` | `fma_audio` + `fma_metadata` |
| `moisesdb_mix` | MoisesDB mixtures (1,998 windows of 10 s) | `clips` | `moisesdb_mix` |
| `moisesdb_stems` | MoisesDB stems (1,546 windows of 10 s) | `clips` | `moisesdb_stems` |
| `musiccaps` | MusicCaps (964 clips of 10 s) | `clips` | `musiccaps` |
| `song_describer` | Song Describer (8,364 chunks of 10 s) | `clips` | `song_describer` |

The windowing of MoisesDB and Song Describer into 10 s clips is not part of this repository.

```bash
# once per clip set: reference embeddings and FAD statistics (written into the set's folder)
python -m evaluation.build_references dataset=musiccaps

# SAGE (checkpoint: paths.sage_checkpoint, or checkpoint=<file>)
python -m evaluation.reconstruction dataset=fma model=sage
python -m evaluation.reconstruction dataset=musiccaps model=sage
# a baseline: model=same | same-s | sao-vae | codicodec | music2latent
python -m evaluation.reconstruction dataset=musiccaps model=same-s
```

Results go to `<paths.eval_output>/recon_<dataset>/<model>/metrics/` (default
`results/`). On FMA the target embeddings are cached in `paths.eval_cache` and shared by
every model, so their FADs use the same targets. All settings are in
`configs/reconstruction.yaml`.

SAGE is evaluated as in the paper: sampled latent (`deterministic=true` decodes the
posterior mean instead) and variable-length attention `tri2`. Three parts of the
evaluation are random: SAGE's sampled latent, SAME's decoding noise, and the 10 s crop that
LAION-CLAP takes of longer audio for FAD-CLAP (FMA clips are 30 s). All are seeded per file
(`seed=`, default 0), so a run is reproducible and independent of how it is sharded or
resumed; the paper's numbers carry the noise of one unseeded draw. A run resumed into
the same output folder must keep its settings (they are recorded in `metrics/parts/run.json`).
`compute_ms_metrics=true` adds the stereo-image metrics (width bias, width distance, SI-SDR of
Side and Mid) used in Table 7.

## Semantic probing (Tables 4 and 9)

```bash
python -m evaluation.maeb encoder=sage        # the 19 tasks: FMA (6) + MoisesDB (7) + upstream MAEB music (6)
python -m evaluation.maeb encoder=same-s      # a baseline
python -m evaluation.maeb encoder=clap        # CLAP oracle row (checkpoint: paths.clap_teacher)
```

`suite=fma | moisesdb | fma_moisesdb | maeb_music` runs a subset; results go to
`<paths.eval_output>/maeb/<name>/`. All settings are in `configs/maeb.yaml`.

## Tables

```bash
python -m evaluation.tables --recon "FMA test=results/recon_fma" --recon "MoisesDB mixtures=results/recon_moisesdb_mix" \
    --maeb results/maeb/SAGE_FTe992 --maeb results/maeb/same-s --maeb results/maeb/clap
```

Not reproduced by this repository: the inference timings of Tables 1 and 8, the MUSHRA
test of Table 3, and the ablations and sweeps of Tables 5, 7 (phase-1 checkpoints with and
without L_SD), 10 and 11.
