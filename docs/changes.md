# Changes from the development code

For readers who used the pre-release code of the paper. Checkpoints of the paper load
unchanged: their saved class paths are mapped to the new ones (`sage/compat.py`), and a
resumed run drops the one buffer that no longer exists.

## Packages and entry points

- Everything lives in the `sage` package (`sage.model` for the architecture, `sage.nn`
  for building blocks, `sage.training`, `sage.inference`); `SAGE.from_checkpoint`
  replaces `EuleroEncodeDecode`.
- The Hydra config directory is `configs/`; the two training phases are the recipes
  `+experiment=pretrain` and `+experiment=decoder_ft`.
- The per-model evaluation scripts are one package: `python -m evaluation.reconstruction`,
  `build_references`, `maeb`, `tables`, configured with Hydra (`dataset=... model=...`)
  instead of command-line flags.

## Paths and defaults

- The repository-root `config.py` and its `.env` are gone. Paths are set in
  `configs/paths/default.yaml` through environment variables: `DATA_PATH` is now
  `FMA_AUDIO`; `FAST` and `WORK` are no longer read (use `SAGE_MODELS`, `CLAP_TEACHER_CKPT`,
  `CLAP_FAD_CKPT`, `PANN_CKPT`, `SAO_VAE_DIR`); the MoisesDB, MusicCaps and Song Describer
  clip sets have their own variables.
- The CLAP teacher checkpoint is a loss setting,
  `trainer.loss_config.semantic_distill.teacher_checkpoint` (default `paths.clap_teacher`).
- `trainer.device` defaults to `auto`, the W&B project to `sage`, and Hydra no longer
  changes the working directory (`runs/` is created where the command is launched).
- Resuming no longer needs a SLURM script: launching a run name again continues its newest
  checkpoint (`runs/<name>/*/`, including Lightning's save on a requeue) in its own folder and
  its W&B run (`.run_ids/<name>`), which the paper's scripts did with `WANDB_RUN_ID` and
  `+ckpt_path`. `auto_resume=false` starts over.
- Without a recipe, `configs/trainer.yaml` now holds the pretraining weights of Table 6
  (mel 0.5, KL 1e-4, sum-and-difference 1; previously 0.3, 1e-3 and 0).

## Training

- The loss manager builds only the seven terms of the paper and rejects other loss blocks.
  The other losses of the development code are in `sage/nn/losses/experimental/` (no longer
  exported by `sage.nn.losses`) and are added with `trainer.loss_config.extra`.
- Schedules and the semantic gate count generator updates (`gen_step`, saved in the
  checkpoint). The paper's gate of 25,000 optimizer steps is `detach_warmup_steps: 8334`;
  a pre-release config that still says 25000 opens the gate about three times later.
- `+init_from=` loads the weights before `torch.compile`, and `ema_decay` is read from the
  config.

## Evaluation

- SAGE and every baseline are scored by the same code (`evaluation/codecs.py`).
- The random parts (SAGE's sampled latent, SAME's decoding noise, the 10 s crop of LAION-CLAP
  for FAD-CLAP) are seeded per file (`seed=`, default 0). The development evaluators drew
  them from an unseeded generator, so their numbers differ from any seeded run by that noise.
- MAEB runs the 19 tasks of the paper in one call by default (`suite=paper`).
- Removed: inference timings, the latent UMAP figure, the results notebook, backfill flags.
