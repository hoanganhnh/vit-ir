# SatLetter: Satellite Letter-Shape Image Retrieval

> Ứng dụng Vision Transformers (IRT) cho truy xuất ảnh vệ tinh chứa hình dạng chữ cái.

**Based on**: [Training Vision Transformers for Image Retrieval](paper/) (El-Nouby et al., ICML 2022)

## Mô tả

Hệ thống **cross-domain image retrieval** sử dụng DeiT-Small Vision Transformer:

- **Input**: Ảnh chữ cái (viết tay EMNIST / rendered font / ảnh vệ tinh mẫu)
- **Output**: Top-K ảnh vệ tinh có hình dạng tự nhiên tương tự nhất

**Ví dụ**: Input ảnh chữ "S" → Trả về ảnh vệ tinh con sông uốn cong hình chữ S.

## Kiến trúc

```
Input Image (224×224)
    │
    ▼
DeiT-Small Backbone (22M params)
    │
    ▼
CLS Token → L2-Normalize → 384-D Descriptor
    │
    ▼
Cosine Similarity → Top-K Results
```

**Training**: Contrastive Loss (β=0.5) + KoLeo Regularization (λ=0.7) + Cross-Batch Memory

## Cấu trúc dự án

```
vit-ir/
├── src/                            # Core library
│   ├── models/backbone.py          # IRTModel (DeiT-Small + GeM pooling)
│   ├── losses/irt_losses.py        # Contrastive + KoLeo + XBM
│   ├── data/datasets.py            # DataLoaders + CrossDomainDataset
│   └── utils/evaluation.py         # Recall@K, mAP
├── scripts/                        # Dataset & training pipeline
│   ├── 01_download_emnist.py       # EMNIST Letters (145K images)
│   ├── 02_generate_rendered_fonts.py # Rendered fonts (15.6K images)
│   ├── 03_scrape_nasa_landsat.py   # NASA satellite letters
│   ├── 04_augment_satellite.py     # Satellite augmentation
│   ├── 05_verify_dataset.py        # Dataset verification
│   ├── run_pipeline.py             # Master pipeline runner
│   ├── train_stage1.sh             # Stage 1: Shape pretraining
│   ├── train_stage2.sh             # Stage 2: Cross-domain fine-tuning
│   └── evaluate_cross_domain.py    # Cross-domain evaluation
├── train.py                        # Main training entry point
├── demo.py                         # Interactive retrieval demo
├── dataset/                        # Generated datasets (gitignored)
├── checkpoints/                    # Trained models (gitignored)
├── notebook/                       # Colab notebooks
├── paper/                          # Reference paper
├── docs/                           # Documentation
└── requirements.txt
```

## Quick Start (Google Colab)

### 1. Setup

```bash
git clone https://github.com/hoanganhnh/vit-ir.git
cd vit-ir
pip install -r requirements.txt
```

### 2. Dataset Pipeline

```bash
# Full pipeline (EMNIST + Fonts + NASA + Verification)
python scripts/run_pipeline.py

# Hoặc từng bước
python scripts/01_download_emnist.py
python scripts/02_generate_rendered_fonts.py
python scripts/03_scrape_nasa_landsat.py
python scripts/04_augment_satellite.py    # Sau khi có satellite images
python scripts/05_verify_dataset.py
```

### 3. Training

```bash
# Stage 1: Shape pretraining (EMNIST, 20 epochs)
bash scripts/train_stage1.sh

# Stage 2: Cross-domain fine-tuning (30 epochs)
bash scripts/train_stage2.sh checkpoints/emnist_best.pth
```

### 4. Evaluation

```bash
python scripts/evaluate_cross_domain.py \
    --checkpoint checkpoints/cross_domain_best.pth \
    --query_dir dataset/sat_fonts \
    --gallery_dir dataset/satellite_letters
```

### 5. Demo

```bash
python demo.py \
    --checkpoint checkpoints/cross_domain_best.pth \
    --query path/to/query.jpg \
    --gallery dataset/satellite_letters \
    --top_k 10
```

## Dataset

| Tier | Dataset | Size | Mục đích |
|------|---------|------|----------|
| 1 | EMNIST Letters | ~145K train / ~14.8K test | Proxy pretraining |
| 2 | Rendered Fonts | ~15.6K images | Bridge domain |
| 2b | Sat Fonts | ~6.2K images | Font overlay trên satellite texture |
| 3 | NASA Satellite | 71 raw → ~994 augmented | Ground truth evaluation |

## Training Pipeline

| Stage | Dataset | Epochs | LR | Backbone |
|-------|---------|--------|-----|----------|
| 1 — Shape Pretrain | EMNIST (145K) | 20 | 3e-5 | Fully trainable |
| 2 — Cross-Domain | EMNIST + Fonts + Satellite (10x) | 30 | 1e-5 | Freeze 6/12 layers |

## Evaluation Metrics

| Metric | Mô tả |
|--------|-------|
| Recall@K (K=1,5,10) | Tỷ lệ queries có ≥1 correct match trong top-K |
| mAP | Mean Average Precision |

## Success Criteria

| Metric | Mục tiêu | Stretch |
|--------|----------|---------|
| Cross-domain R@1 (EMNIST → Satellite) | >40% | >55% |
| Cross-domain R@5 (EMNIST → Satellite) | >65% | >80% |
| Same-domain R@1 (Satellite → Satellite) | >60% | >75% |

## Timeline

| Tuần | Nhiệm vụ |
|------|----------|
| 1 | Thu thập dataset (pipeline + manual Google Earth) |
| 2 | Reproduce IRT, train EMNIST, implement core modules |
| 3 | Fine-tune cross-domain, evaluation |
| 4 | Ablation studies, demo, báo cáo |

## Documentation

| Document | Mô tả |
|----------|-------|
| [Project Overview & PDR](docs/project-overview-pdr.md) | Tổng quan dự án và yêu cầu |
| [System Architecture](docs/system-architecture.md) | Kiến trúc hệ thống chi tiết |
| [Codebase Summary](docs/codebase-summary.md) | Tóm tắt codebase |
| [Code Standards](docs/code-standards.md) | Quy chuẩn code |
| [Project Roadmap](docs/project-roadmap.md) | Lộ trình phát triển |
| [Deployment Guide](docs/deployment-guide.md) | Hướng dẫn triển khai trên Colab |
| [Colab Guide](docs/COLAB_GUIDE.md) | Hướng dẫn chi tiết cho Colab |

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
Pillow>=9.0.0
numpy>=1.21.0
requests>=2.28.0
tqdm>=4.64.0
matplotlib>=3.5.0
```

## License

Đồ án môn học — PTIT
