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

# # Run A — BugFix: window=4 (fixes T=4 < window=8 at stage 4), everything else identical
# Run-Experiment ($AE_BASE + @(
#     "models.model.encoder.window_size=4",
#     "models.model.decoder.window_size=4",
#     "trainer.wandb.name=Swin_real_AE_window_4"
# ))

# # Run B — SwinT: window=4 + depths=[2,2,6,2] (Swin-T depth profile, +2 blocks in stage 3)
# Run-Experiment ($AE_BASE + @(
#     "models.model.encoder.window_size=4",
#     "models.model.decoder.window_size=4",
#     "models.model.encoder.depths=[2,2,6,2]",
#     "models.model.decoder.depths=[2,6,2,2]",
#     "trainer.wandb.name=Swin_real_AE_tiny_window_4"
# ))

# # Run C — 3Stage: 3 stages [2,6,2], window=4 — latent (B,64,64,8) vs (B,64,32,4) baseline
Run-Experiment ($AE_BASE + @(
    "models.model.encoder.depths=[2,6,2]",
    "models.model.encoder.num_heads=[4,8,16]",
    "models.model.encoder.window_size=4",
    "models.model.decoder.window_size=4",
    "models.model.decoder.depths=[2,6,2]",
    "models.model.decoder.num_heads=[16,8,4]",
    "models.model.encoder.embed_dim=64",
    "models.model.decoder.embed_dim=64",
    "trainer.wandb.name=Swin_real_AE_3_stages_embed64_window_4"
))

# Run D — Wider: embed 48→64, num_heads=[4,8,16,32] (head_dim=16 maintained), ~21M params
# Run-Experiment ($AE_BASE + @(
#     "models.model.encoder.embed_dim=64",
#     "models.model.decoder.embed_dim=64",
#     "models.model.encoder.num_heads=[4,8,16,32]",
#     "models.model.decoder.num_heads=[32,16,8,4]",
#     "trainer.wandb.name=Swin_AE_largerPatchEmbed"
# ))

# TODO                             
# Stage dims: [96, 192, 384]
Run-Experiment ($AE_BASE + @(
    "models.model.encoder.depths=[2,6,2]",
    "models.model.encoder.num_heads=[3,6,12]",
    "models.model.encoder.window_size=4",
    "models.model.decoder.window_size=4",
    "models.model.encoder.embed_dim=96",
    "models.model.decoder.depths=[2,6,2]",
    "models.model.decoder.num_heads=[12,6,3]",
    "models.model.decoder.embed_dim=96",
    "trainer.wandb.name=Swin_real_AE_3stages_embed96_window_4"
))


Write-Host ""
Write-Host "============================================================"
Write-Host "SWEEP COMPLETE: $pass passed, $fail failed."
Write-Host "============================================================"
