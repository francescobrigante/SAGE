# Sweep: 4 runs — bottleneck ablation (default complex, spectral, improper) + loss/KL ablation
# Run 1: Default complex, MRSTFT + MRMEL 0.3, KL 1e-4
# Run 2: Default complex, MRSTFT + MRMEL 0.3, KL 5e-5
# Run 3: Spectral (eigenvectors), MRSTFT + MRMEL 0.3, KL 1e-4
# Run 4: Improper, STFT MSE + MRMEL 0.3, KL 5e-5

$EPOCHS = 80
$KL_ANNEAL = 15
$pass = 0
$fail = 0

function Run-Experiment {
    param([string[]]$RunArgs)
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "RUN: uv run train.py $RunArgs"
    Write-Host "============================================================"
    uv run train.py @RunArgs
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Run completed successfully."
        $script:pass++
    } else {
        Write-Host "[FAILED] Run exited with error - continuing to next run."
        $script:fail++
    }
}


# ============================================================
# Architecture ablation — Swin AE real (SkipBottleneck, STFT MSE only, 80 epochs)
# Baseline (already run): [2,2,4,2], embed=48, window=8  → name: Swin_original_AE
#
# Run A: BugFix    — window 8→4, same depths/embed      → fixes T<window at stage 4
# Run B: SwinT     — window=4 + depths=[2,2,6,2]         → Swin-T depth profile
# Run C: 3Stage    — 3 stages [2,6,2], window=8          → larger latent (64,8), no bug
# Run D: Wider     — embed 48→64, num_heads=[4,8,16,32]  → +75% params, same structure
# ============================================================

$AE_BASE = @(
    "models=swin_real_mini",
    "trainer.optimizer.lr=1e-4",
    "trainer.optimizer.weight_decay=0.05",
    "+trainer.optimizer.weight_decay_exclude_1d=true",
    "trainer.trainer.clip_grad_norm=5.0",
    "trainer.trainer.kl_annealing_epochs=0",
    "trainer.trainer.epochs=$EPOCHS",
    "trainer.loss_config.mrmel.weights.mrmel=0.0",
    "trainer.loss_config.bottleneck.weights.kl=0.0",
    "models.model.parameters_to_predict=1",
    "+models.model.bottleneck.skip_bottleneck=true"
)


# Best run
# Run-Experiment ($AE_BASE + @(
#     "models.model.encoder.depths=[2,6,2]",
#     "models.model.encoder.num_heads=[3,6,12]",
#     "models.model.encoder.window_size=4",
#     "models.model.decoder.window_size=4",
#     "models.model.decoder.depths=[2,6,2]",
#     "models.model.decoder.num_heads=[12,6,3]",
#     "trainer.wandb.name=Swin_real_AE_3_stages_window_4"
# ))

# Refactor validation — same arch as Best run, refactored encoder/decoder modules
Run-Experiment ($AE_BASE + @(
    "models.model.encoder.depths=[2,6,2]",
    "models.model.encoder.num_heads=[3,6,12]",
    "models.model.encoder.window_size=4",
    "models.model.decoder.window_size=4",
    "models.model.decoder.depths=[2,6,2]",
    "models.model.decoder.num_heads=[12,6,3]",
    "trainer.wandb.name=Swin_real_AE_3stage_w4_refactor"
))


# # Run C — 3Stage: 3 stages [2,6,2], window=4 — latent (B,64,64,8) vs (B,64,32,4) baseline
# Run-Experiment ($AE_BASE + @(
#     "models.model.encoder.depths=[2,2,6,2]",
#     "models.model.encoder.num_heads=[3,6,12,24]",
#     "models.model.encoder.window_size=4",
#     "models.model.decoder.window_size=4",
#     "models.model.encoder.embed_dim=72",
#     "models.model.decoder.embed_dim=72",
#     "models.model.decoder.depths=[2,6,2,2]",
#     "models.model.decoder.num_heads=[24,12,6,3]",
#     "trainer.wandb.name=Swin_real_AE_window_4_embed72"
# ))


# # TODO                             
# # Stage dims: [96, 192, 384]
# Run-Experiment ($AE_BASE + @(
#     "models.model.encoder.depths=[2,6,2]",
#     "models.model.encoder.num_heads=[3,6,12]",
#     "models.model.encoder.window_size=4",
#     "models.model.decoder.window_size=4",
#     "models.model.encoder.embed_dim=96",
#     "models.model.decoder.embed_dim=96",
#     "models.model.decoder.depths=[2,6,2]",
#     "models.model.decoder.num_heads=[12,6,3]",
#     "trainer.wandb.name=Swin_real_AE_3stages_embed96_window_4"
# ))

$VAE_BASE = @(
    "models=swin_real_mini",
    "trainer.optimizer.lr=1e-4",
    "trainer.optimizer.weight_decay=0.05",
    "+trainer.optimizer.weight_decay_exclude_1d=true",
    "trainer.trainer.clip_grad_norm=5.0",
    "trainer.trainer.epochs=$EPOCHS",
    "trainer.loss_config.mrmel.weights.mrmel=0.0",
    "models.model.parameters_to_predict=2",
    "trainer.trainer.kl_annealing_epochs=20",
    "trainer.loss_config.bottleneck.weights.kl=1e-4"
)

# Run-Experiment ($VAE_BASE + @(
#     "models.model.encoder.depths=[2,6,2]",
#     "models.model.encoder.num_heads=[3,6,12]",
#     "models.model.encoder.window_size=4",
#     "models.model.decoder.window_size=4",
#     "models.model.decoder.depths=[2,6,2]",
#     "models.model.decoder.num_heads=[12,6,3]",
#     "trainer.wandb.name=Swin_real_VAE_3stage_KL1e-4_window_4"
# ))



# ============================================================
# Bottleneck ablation — Swin CVAE complex (swin_cvae_mini, 80 epochs)
# Requires data overrides: cac=false, model_channels=2
# ============================================================

$CVAE_BASE = @(
    "models=swin_cvae_mini",
    "data.train_dataset.cac=false",
    "data.eval_dataset.cac=false",
    "data.demo.istft_params.cac=false",
    "data.model_channels=2",
    "trainer.optimizer.lr=1e-4",
    "trainer.optimizer.weight_decay=0.05",
    "+trainer.optimizer.weight_decay_exclude_1d=true",
    "trainer.trainer.clip_grad_norm=5.0",
    "trainer.trainer.epochs=$EPOCHS",
    "trainer.trainer.kl_annealing_epochs=$KL_ANNEAL",
    "trainer.loss_config.mrstft.weights.mrstft=0.3",
    "trainer.loss_config.mrmel.weights.mrmel=0.3"
)

# Run 1: Default improper (KL barrier), KL 1e-4
Run-Experiment ($CVAE_BASE + @(
    "trainer.loss_config.bottleneck.weights.kl=1e-4",
    "trainer.wandb.name=CVAE_default_KL1e-4"
))

# Run 2: Default improper, KL 5e-5
Run-Experiment ($CVAE_BASE + @(
    "trainer.loss_config.bottleneck.weights.kl=5e-5",
    "trainer.wandb.name=CVAE_default_KL5e-5"
))

# Run 3: Spectral parameterization (eigendecomposition, no KL clamp), KL 1e-4
Run-Experiment ($CVAE_BASE + @(
    "models.model.bottleneck.apply_spectral_parameterization=true",
    "models.model.parameters_to_predict=3",
    "trainer.loss_config.bottleneck.weights.kl=1e-4",
    "trainer.wandb.name=CVAE_spectral_KL1e-4"
))

# Run 4: Improper + STFT MSE only (no MRSTFT), KL 5e-5
Run-Experiment ($CVAE_BASE + @(
    "trainer.loss_config.mrstft.weights.mrstft=0.0",
    "trainer.loss_config.stft_mse.weights.stft_mse=1.0",
    "trainer.loss_config.bottleneck.weights.kl=5e-5",
    "trainer.wandb.name=CVAE_improper_stftmse_KL5e-5"
))

Write-Host ""
Write-Host "============================================================"
Write-Host "SWEEP COMPLETE: $pass passed, $fail failed."
Write-Host "============================================================"
