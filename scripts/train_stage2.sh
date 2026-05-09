#!/bin/bash
# Stage 2: Cross-Domain Fine-tuning
# Uses mixed batches: EMNIST + Fonts + Satellite (oversampled)
# Requires Stage 1 checkpoint

STAGE1_CKPT="${1:-checkpoints/emnist_best.pth}"

if [ ! -f "$STAGE1_CKPT" ]; then
    echo "ERROR: Stage 1 checkpoint not found: $STAGE1_CKPT"
    echo "Run scripts/train_stage1.sh first"
    exit 1
fi

echo "=== Stage 2: Cross-Domain Fine-tuning ==="
echo "Pretrained from: $STAGE1_CKPT"

python train.py \
    --dataset cross_domain \
    --pretrained_from "$STAGE1_CKPT" \
    --epochs 30 \
    --batch_size 64 \
    --lr 1e-5 \
    --freeze_layers 6 \
    --lambda_koleo 0.7 \
    --margin 0.5 \
    --xbm_size 2048 \
    --satellite_oversample 10 \
    --eval_every 5 \
    --save_every 5 \
    "${@:2}"

echo "Stage 2 complete. Best checkpoint: checkpoints/cross_domain_best.pth"
echo "Run demo: python demo.py --checkpoint checkpoints/cross_domain_best.pth --query <letter_image> --gallery dataset/satellite_letters"
