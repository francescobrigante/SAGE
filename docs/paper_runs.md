# Paper runs

Every result of the paper and the command that reproduces it. Commands run from the
repository root in the environment of `uv sync --extra train --extra eval` (add
`--extra baselines` for the baseline rows). On a SLURM cluster the same commands go
through the templates in `scripts/slurm/` (`train.sbatch <recipe>`,
`eval.sbatch <script> <args>`; pass `-A <account> -p <partition>` to `sbatch`).

## Training (Table 6)

| Result | Command |
|---|---|
| Phase 1, pretraining (500 epochs, 16 GPUs, global batch 128) | `python train.py +experiment=pretrain trainer.trainer.num_gpus=4 ++trainer.trainer.num_nodes=4` |
| Phase 2, decoder fine-tuning → `SAGE_FTe992.ckpt` (epoch 992) | `python train.py +experiment=decoder_ft +init_from=<phase-1 checkpoint> trainer.trainer.num_gpus=4 ++trainer.trainer.num_nodes=4` |

The recipes (`configs/experiment/pretrain.yaml`, `decoder_ft.yaml`) hold every value of
Table 6; `tests/test_recipes.py` checks them. The batch size is global, so fewer GPUs run
the same recipe.

## Reconstruction (Table 2)

Five evaluation sets, two protocols:

| Set | Protocol | Data |
|---|---|---|
| FMA test (11,263 clips of 30 s) | `fma` | `fma_large/` + `fma_metadata/tracks.csv` |
| MoisesDB mixtures (1,998 windows of 10 s) | `clips` | a folder of 10 s mixture windows |
| MoisesDB stems (1,546 windows of 10 s) | `clips` | a folder of 10 s stem windows |
| MusicCaps (964 clips of 10 s) | `clips` | 44.1 kHz stereo clips |
| Song Describer (8,364 chunks of 10 s) | `clips` | 44.1 kHz stereo 10 s chunks |

The windowing of MoisesDB and Song Describer into 10 s clips is not part of this repository.

```bash
# once per clip set: reference embeddings and FAD statistics (written into the set's folder)
python -m evaluation.build_references --data-dir <clip set>

# SAGE on FMA; share --cache-dir across models so their FADs use the same targets
python -m evaluation.reconstruction --model sage --checkpoint SAGE_FTe992.ckpt --protocol fma \
    --data-dir <fma_large> --fma-csv <fma_metadata/tracks.csv> --cache-dir <cache> --output-dir runs/recon_fma
# SAGE on a clip set
python -m evaluation.reconstruction --model sage --checkpoint SAGE_FTe992.ckpt --protocol clips \
    --data-dir <clip set> --output-dir runs/recon_<set>
# a baseline: --model same | same-s | sao-vae | codicodec | music2latent (no --checkpoint)
python -m evaluation.reconstruction --model same-s --protocol clips --data-dir <clip set> --output-dir runs/recon_<set>
```

SAGE is evaluated as in the paper: sampled latent (`--deterministic` decodes the
posterior mean instead) and variable-length attention `tri2`. Three parts of the
evaluation are random: SAGE's sampled latent, SAME's decoding noise, and the 10 s crop that
LAION-CLAP takes of longer audio for FAD-CLAP (FMA clips are 30 s). All are seeded per file
(`--seed`, default 0), so a run is reproducible and independent of how it is sharded or
resumed; the paper's numbers carry the noise of one unseeded draw. A run resumed into
the same output folder must keep its settings (they are recorded in `metrics/parts/run.json`).
`--compute-ms-metrics` adds the stereo-image metrics (width bias, width distance, SI-SDR of
Side and Mid) used in Table 7.

## Semantic probing (Tables 4 and 9)

```bash
python -m evaluation.maeb --encoder sage --checkpoint SAGE_FTe992.ckpt --with-moisesdb             # FMA + MoisesDB (13)
python -m evaluation.maeb --encoder sage --checkpoint SAGE_FTe992.ckpt --maeb-original-music-only  # upstream MAEB (6)
python -m evaluation.maeb --encoder same-s --with-moisesdb                                         # a baseline
python -m evaluation.maeb --encoder clap --checkpoint music_audioset_epoch_15_esc_90.14.pt         # CLAP oracle row
```

## Tables

```bash
python -m evaluation.tables --recon "FMA test=runs/recon_fma" --recon "MoisesDB mixtures=runs/recon_moisesdb_mix" \
    --maeb maeb_results/SAGE_FTe992 --maeb maeb_results/same-s
```

Not reproduced by this repository: the inference timings of Tables 1 and 8, the MUSHRA
test of Table 3, and the ablations and sweeps of Tables 5, 7 (phase-1 checkpoints with and
without L_SD), 10 and 11.
