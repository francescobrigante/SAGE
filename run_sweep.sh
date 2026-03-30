#!/bin/bash
# Overnight sweep: 2 complex runs (Mode A, 80 epochs) — loss function ablation
# Run 1: MRSTFT only  |  Run 2: MRSTFT + MRMEL

EPOCHS=80
PASS=0
FAIL=0

run() {
    echo ""
    echo "============================================================"
    echo "RUN: uv run train.py $*"
    echo "============================================================"
    uv run train.py "$@"
    if [ $? -eq 0 ]; then
        echo "[OK] Run completed successfully."
        ((PASS++))
    else
        echo "[FAILED] Run exited with error — continuing to next run."
        ((FAIL++))
    fi
}

# Run 1: MRSTFT only
run +experiment=cplx trainer.trainer.epochs=$EPOCHS trainer.wandb.name=TEST1 \
    trainer.loss_config.spectral.weights.stft_mse=0.0 \
    trainer.loss_config.spectral.weights.mrstft=1.0

# Run 2: MRSTFT + MRMEL
run +experiment=cplx trainer.trainer.epochs=$EPOCHS trainer.wandb.name=TEST2 \
    trainer.loss_config.spectral.weights.stft_mse=0.0 \
    trainer.loss_config.spectral.weights.mrstft=1.0 \
    trainer.loss_config.mrmel.weights.mrmel=1.0

echo ""
echo "============================================================"
echo "SWEEP COMPLETE: $PASS passed, $FAIL failed."
echo "============================================================"
