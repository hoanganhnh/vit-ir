#!/bin/bash
# Stage 1: Shape Pretraining on EMNIST + Rendered Fonts (merged)
# Run on combined dataset to learn letter shape recognition
# Expected: ~65K images, 20 epochs

echo "=== Stage 1: Shape Pretraining ==="
echo "Dataset: emnist (will add cross_domain support for merged in future)"

python train.py \
    --dataset emnist \
    --epochs 20 \
    --batch_size 64 \
    --lr 3e-5 \
    --lambda_koleo 0.7 \
    --margin 0.5 \
    --eval_every 5 \
    --save_every 5 \
    "$@"

echo "Stage 1 complete. Best checkpoint: checkpoints/emnist_best.pth"
echo "Next: run scripts/train_stage2.sh"
