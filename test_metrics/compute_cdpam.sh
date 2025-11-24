#!/bin/bash

python compute_cdpam.py \
  --ref_dir /home/cerovaz/repos/data/jamendo_full/test_trimmed \
  --pred_dir /home/cerovaz/repos/ICML/Eulero_BackBone/runs/inference/real_dataset_outputs_24epoch \
  --pattern "*.mp3" \
  --pred_ext wav \
  --out cdpam_results.csv
