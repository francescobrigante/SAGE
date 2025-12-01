# Trainer Configuration Guide

## Purpose
This guide describes how to configure optimisation, scheduling, and logging for EulerAudioBackbone training runs. The training stack relies on Hydra for composition, so each concern (data, model, trainer) lives in a dedicated configuration group.

## File Layout
```
conf/
  config.yaml          # Hydra defaults list
  data/                # Dataset & dataloader definitions
  model/               # Autoencoder variants
  trainer/             # Optimisation, logging, runtime settings
```
Select alternate variants with `python train.py trainer=<name> data=<name> model=<name>` or override individual keys (`python train.py trainer.optimizer.config.lr=3e-4`).

## Key Trainer Settings
| Path | Description |
| --- | --- |
| `trainer.seed` | Global seed used for RNG initialisation (PyTorch, NumPy, Python). |
| `trainer.strategy` | PyTorch Lightning strategy (`auto`, `ddp`, `deepspeed`, ...). |
| `trainer.num_gpus` | Number of GPUs to request; set to `0` for CPU-only. |
| `trainer.precision` | Mixed-precision policy (`16-mixed`, `bf16-mixed`, etc.). |
| `trainer.trainer.max_epochs` / `max_steps` | Training horizon. |
| `trainer.trainer.gradient_clip_val` | Gradient clipping threshold. |
| `trainer.trainer.accumulate_grad_batches` | Gradient accumulation factor. |
| `trainer.trainer.save_every_n_epochs` | How often checkpoints are saved. |
| `trainer.trainer.deterministic` | To activate CUDA deterministic algorithms. Doesn't with standard models in this repo, due to padding.|
| `trainer.trainer.strict_deterministic` | To activate CUDA deterministic algorithms. Doesn't with standard models in this repo due to padding. |

Note: We didn't specify the parameters that have the same name as the pl Trainer such as `num_sanity_val_steps`.  

### Optimiser Configuration
Define the optimiser under `trainer.optimizer`:
```yaml
optimizer:
  type: AdamW
  config:
    lr: 3e-4
    betas: [0.9, 0.999]
    weight_decay: 1e-4
```
Swap optimisers by changing `type`. Any keyword arguments supported by the underlying torch optimiser can be added to `config`.

### Scheduler Configuration
Schedulers live under `trainer.scheduler`:
```yaml
scheduler:
  type: InverseLR
  config:
    inv_gamma: 1.0
    power: 1.0
    warmup: 0
```
Set `enabled: false` to disable scheduling. When active, the trainer attaches the scheduler to the optimiser according to the semantics of the selected strategy.

### Loss Configuration
Losses are grouped under `trainer.loss_config`. Each entry declares a named loss module with optional configuration and assigns a weight:
```yaml
loss_config:
  spectral:
    stft_mse:
      config:
        reduction: mean
    weights:
      stft_mse: 1.0
```
Add or remove terms to reflect the experiment at hand (e.g., multi-resolution STFT, time-domain L1, adversarial discriminators). Ensure weights remain positive; zero-valued terms are effectively disabled.

#### Bottleneck KL regulariser
When the model uses `VAEBottleneck`, the encoder returns a batch-level KL statistic (`loss_info["kl"]`). To make it part of the generator objective you must add the dedicated weight under `loss_config.bottleneck.weights.kl`:
```yaml
loss_config:
  bottleneck:
    weights:
      kl: 1.0
```
The engine automatically wires this weight into a `ValueLoss(key="kl")`, so no further code changes are required. Other bottleneck types simply ignore the section.

##### Eval Losses

Sets the losses used in eval
```yaml
eval_loss_config:
  sisdr:
    zero_mean: true
    reduction: mean
  stft:
    sample_rate: 44100
    fft_size: 2048
    hop_size: 512
    win_length: 2048
  mel:
    fft_size: 2048
    hop_size: 512
    win_length: 2048
    n_mels: 128
```

### Logging & Checkpointing
| Setting | Purpose |
| --- | --- |
| `trainer.logger` | Configure WandB, TensorBoard, or disable logging. |
| `trainer.wandb.use_wandb` | Convenience flag to toggle WandB integration. |
| `trainer.trainer.check_val_every_n_epoch` | Validation frequency. |
| `trainer.trainer.log_every_n_steps` | Training log interval. |
| `trainer.trainer.save_top_k` / `save_every_n_epochs` | Checkpoint retention policy. |
| `trainer.trainer.log_model_structure` | The model structure will be printed in the terminal. |
| `trainer.trainer.ckpt_dir` | Where to save checkpoints. |
| `trainer.trainer.profile` | If you want to use profiler. Generally turned off. |
| `trainer.trainer.profiler_dir` | Where to save profiler outputs |


## Command-Line Overrides
Hydra allows dot-notation overrides for any value. Examples:
- Adjust batch size and epochs:
  ```bash
  python train.py data.train_dataloader.batch_size=32 trainer.trainer.max_epochs=100
  ```
- Disable WandB logging:
  ```bash
  python train.py trainer.wandb.use_wandb=false 
  ```
- Switch to an alternate model and dataset profile:
  ```bash
  python train.py model=SEANet_real data=jamendo_highres
  ```

## Reproducibility Checklist
1. Fix `trainer.seed` and record it when publishing results.
2. Keep STFT settings identical across training and evaluation configs.
3. Log configuration files alongside checkpoints (Hydra automatically stores them in the run directory).
4. For distributed runs, set deterministic flags (`trainer.deterministic=true`) if required, acknowledging potential performance trade-offs.
5. Version-control environment details (Python version, dependency hashes) to enable exact reruns.

## Troubleshooting
- **Diverging losses:** start with conservative learning rates, verify normalisation layers are calibrated for complex inputs, and ensure gradient clipping is enabled for unstable regimes.
- **Out-of-memory errors:** reduce batch size, enable gradient accumulation, or switch to chunked spectrogram processing.
- **Inconsistent validation metrics:** confirm that validation datasets mirror the training STFT configuration and that `model.autoencoder.pre_transform` is applied consistently.
