# Deployment Guide

## Environment: Google Colab

SatLetter được thiết kế chạy trên Google Colab với GPU T4. Hướng dẫn này cover toàn bộ quy trình từ setup đến inference.

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Google Account | Để truy cập Colab |
| Google Drive | Lưu trữ datasets và checkpoints |
| GPU Runtime | T4 (free tier) hoặc A100 (Colab Pro) |
| Disk Space | ~5GB cho datasets + ~500MB cho checkpoints |

## Quick Start

### 1. Clone Repository

```bash
# Trong Colab notebook
!git clone https://github.com/hoanganhnh/vit-ir.git
%cd vit-ir
```

### 2. Install Dependencies

```bash
!pip install -r requirements.txt
```

### 3. Run Dataset Pipeline

```bash
# Full pipeline: download EMNIST + generate fonts + scrape NASA + verify
!python scripts/run_pipeline.py

# Hoặc từng bước:
!python scripts/01_download_emnist.py
!python scripts/02_generate_rendered_fonts.py
!python scripts/03_scrape_nasa_landsat.py
!python scripts/04_augment_satellite.py  # Sau khi có satellite images
!python scripts/05_verify_dataset.py
```

### 4. Train Model

```bash
# Stage 1: Shape pretraining (EMNIST, ~20 epochs)
!bash scripts/train_stage1.sh

# Stage 2: Cross-domain fine-tuning (30 epochs)
!bash scripts/train_stage2.sh checkpoints/emnist_best.pth
```

### 5. Evaluate

```bash
# Cross-domain evaluation
!python scripts/evaluate_cross_domain.py \
    --checkpoint checkpoints/cross_domain_best.pth \
    --query_dir dataset/sat_fonts \
    --gallery_dir dataset/satellite_letters
```

### 6. Demo

```bash
!python demo.py \
    --checkpoint checkpoints/cross_domain_best.pth \
    --query path/to/query.jpg \
    --gallery dataset/satellite_letters \
    --top_k 10 \
    --save results.png
```

## Google Drive Integration

### Save/Load Datasets

```python
from google.colab import drive
drive.mount('/content/drive')

# Save datasets to Drive
!cp -r dataset/ /content/drive/MyDrive/vit-ir-dataset/

# Load datasets from Drive
!cp -r /content/drive/MyDrive/vit-ir-dataset/ dataset/
```

### Save/Load Checkpoints

```python
# Save checkpoints after training
!cp -r checkpoints/ /content/drive/MyDrive/vit-ir-checkpoints/

# Restore checkpoints
!cp -r /content/drive/MyDrive/vit-ir-checkpoints/ checkpoints/
```

## Colab-Specific Setup

Sử dụng `scripts/colab_setup.py` cho automated setup:

```bash
!python scripts/colab_setup.py
```

Script này sẽ:
1. Detect GPU và set optimal `num_workers`
2. Mount Google Drive (nếu available)
3. Download/restore datasets
4. Verify environment

## Training Configuration

### Recommended Settings (T4 GPU)

| Parameter | Stage 1 | Stage 2 |
|-----------|---------|---------|
| Batch size | 64 | 64 |
| Learning rate | 3e-5 | 1e-5 |
| Epochs | 20 | 30 |
| num_workers | auto (2 on Colab) | auto |
| XBM size | 0 | 2048 |
| Freeze layers | 0 | 6 |

### Training Time Estimates (T4)

| Dataset | Epochs | Est. Time |
|---------|--------|-----------|
| EMNIST (145K) | 20 | ~2-3 hours |
| Cross-domain (merged) | 30 | ~4-5 hours |

## Checkpoint Management

### Checkpoint Format

```python
{
    "epoch": int,
    "model_state_dict": OrderedDict,
    "optimizer_state_dict": OrderedDict,
    "metrics": {"R@1": float, "R@5": float, "R@10": float, "mAP": float},
    "args": dict,  # Training configuration
}
```

### Expected Checkpoints

| File | Size | Description |
|------|------|-------------|
| `emnist_best.pth` | ~85MB | Best Stage 1 model |
| `cross_domain_best.pth` | ~85MB | Best Stage 2 model |
| `*_history.json` | ~10KB | Training metrics log |

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| CUDA OOM | Giảm `batch_size` xuống 32 hoặc 16 |
| NaN loss | KoLeo gradient instability — auto-skipped, kiểm tra learning rate |
| Slow DataLoader | Giảm `num_workers` (Colab chỉ có 2 vCPUs) |
| Dataset not found | Chạy `scripts/run_pipeline.py` trước |
| Pin memory warning | Auto-disabled khi không có CUDA |

### GPU Memory Usage (T4, 15GB)

| Configuration | Approx. Usage |
|--------------|---------------|
| Batch 64, DeiT-Small | ~4GB |
| + XBM 2048 | ~5GB |
| + Gradient accumulation | ~6GB |
| Peak during eval | ~8GB |
