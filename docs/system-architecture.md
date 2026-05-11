# System Architecture

## Overview

SatLetter sử dụng kiến trúc IRT (Image Retrieval with Transformers), áp dụng DeiT-Small Vision Transformer để trích xuất image descriptors cho bài toán truy xuất ảnh vệ tinh theo hình dạng chữ cái.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    SatLetter Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐   ┌──────────────┐   ┌───────────────────┐    │
│  │  Input   │──▶│  DeiT-Small  │──▶│  Feature Vector   │    │
│  │ 224×224  │   │   Backbone   │   │  384-D (L2-norm)  │    │
│  └─────────┘   │   22M params │   └────────┬──────────┘    │
│                 └──────────────┘            │               │
│                                            ▼               │
│                                   ┌────────────────┐       │
│                                   │ Cosine Sim     │       │
│                                   │ (query×gallery)│       │
│                                   └────────┬───────┘       │
│                                            ▼               │
│                                   ┌────────────────┐       │
│                                   │   Top-K        │       │
│                                   │   Results      │       │
│                                   └────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Model — IRTModel (`src/models/backbone.py`)

| Component | Details |
|-----------|---------|
| Backbone | DeiT-Small (22M params, 384-D output) via `timm` |
| Pooling | CLS token (default), Average, Max, hoặc GeM (learnable) |
| Normalization | L2-normalize output descriptors |
| Projection | Optional linear projection head (384-D → custom dim) |
| Output | Compact image descriptor vector cho cosine similarity |

**Variants theo paper:**
- **IRTO**: Chỉ contrastive loss (baseline)
- **IRTL**: Contrastive + distillation head
- **IRTR**: Contrastive + KoLeo regularization (variant chính)

**GeM Pooling:**
```
GeM(x) = (mean(x^p))^(1/p)
```
Learnable exponent `p` (default=3.0). p=1 → average pooling, p→∞ → max pooling.

### 2. Loss Functions (`src/losses/irt_losses.py`)

#### Contrastive Loss
```
L = (1/N) Σ_i [ Σ_{j:y_i=y_j} [1 - z_i·z_j]₊
               + Σ_{j:y_i≠y_j} [z_i·z_j - β]₊ ]
```
- Positive pairs: minimize distance (cùng class)
- Negative pairs: push beyond margin β (khác class)
- Supports Cross-Batch Memory (XBM) cho thêm negative pairs

#### KoLeo Loss (Kozachenko-Leonenko)
```
L_KoLeo = -(1/N) Σ_i log(ρ_i)
ρ_i = min_{j≠i} ||z_i - z_j||
```
- Maximize khoảng cách nearest-neighbor → phân bố đều trên hypersphere
- Chống feature collapse
- Nguồn: Sablayrolles et al. (ICLR 2019)

#### Combined IRT Loss
```
L = L_contrastive + λ × L_KoLeo    (λ = 0.7)
```

#### Cross-Batch Memory (XBM)
- FIFO queue lưu embeddings từ batches trước
- Cung cấp thêm negative pairs cho contrastive learning
- Nguồn: Wang et al. (CVPR 2020)

### 3. Dataset Pipeline (`src/data/datasets.py`)

#### 3-Tier Dataset Strategy

```
Tier 1: EMNIST Letters     ──▶  Shape pretraining (145K images)
         │
Tier 2: Rendered Fonts     ──▶  Bridge domain (15.6K images)
        + Sat Fonts              Synthetic satellite fonts (6.2K images)
         │
Tier 3: NASA Satellite     ──▶  Evaluation only (71 raw → 994 augmented)
```

| Tier | Dataset | Size | Mục đích |
|------|---------|------|----------|
| 1 | EMNIST Letters | ~145K train / ~14.8K test | Proxy pretraining — nhận biết hình dạng chữ cái |
| 2 | Rendered Fonts | ~15.6K images | Bridge domain gap |
| 2b | Sat Fonts | ~6.2K images | Font overlay trên satellite texture |
| 3 | NASA Satellite | 71 raw → ~994 augmented | Ground truth evaluation |

#### CrossDomainDataset

Merge nhiều ImageFolder datasets với:
- **Unified labels**: A-Z (0-25) mapping nhất quán
- **Oversampling**: Dataset nhỏ (satellite) được lặp N lần
- **Transparent interface**: Returns `(image, label)` — compatible với training loop

#### Data Transforms

| Mode | Transforms |
|------|-----------|
| Training | Resize(256) → RandomCrop(224) → RandomHorizontalFlip → Normalize(ImageNet) |
| Evaluation | Resize(224) → Normalize(ImageNet) |

### 4. Training Pipeline (`train.py`)

#### Stage 1 — Shape Pretraining (20 epochs)

| Parameter | Value |
|-----------|-------|
| Dataset | EMNIST (145K) hoặc merged EMNIST + Fonts |
| Loss | Contrastive (β=0.5) + KoLeo (λ=0.7) |
| Backbone | DeiT-Small, fully trainable |
| Optimizer | AdamW, lr=3e-5, weight_decay=5e-4 |
| Batch size | 64 |
| XBM | Disabled |

#### Stage 2 — Cross-Domain Fine-tuning (30 epochs)

| Parameter | Value |
|-----------|-------|
| Dataset | CrossDomainDataset: EMNIST + Fonts + Satellite (oversample 10x) |
| Loss | Contrastive (β=0.5) + KoLeo (λ=0.7) |
| Backbone | Freeze 6/12 transformer blocks |
| Optimizer | AdamW, lr=1e-5, weight_decay=5e-4 |
| Batch size | 64 |
| XBM size | 2048 |
| Pretrained | Từ Stage 1 checkpoint |

**NaN Guard**: Training loop tự động skip batches có NaN loss và áp dụng gradient clipping (max_norm=1.0).

### 5. Evaluation Framework (`src/utils/evaluation.py`)

| Metric | Mô tả |
|--------|-------|
| Recall@K (K=1,5,10) | Fraction queries có ≥1 correct match trong top-K |
| mAP | Mean Average Precision trên toàn bộ ranking |

**Evaluation Modes:**

| Mode | Query | Gallery | Mục đích |
|------|-------|---------|----------|
| Cross-domain (primary) | EMNIST/Fonts test | Satellite test | Main use case |
| Self-retrieval | Satellite test | Satellite test | Baseline quality |
| In-domain | Same-domain test | Same-domain test | Sanity check |

### 6. Demo (`demo.py`)

CLI tool cho interactive retrieval:
1. Load trained model từ checkpoint
2. Build gallery index (extract all features)
3. Extract query feature từ single image
4. Cosine similarity ranking → Top-K results
5. Matplotlib visualization (optional)

## Data Flow

```
[Dataset Scripts] → [dataset/] → [DataLoaders] → [IRTModel] → [Embeddings]
                                                                    │
                                                     ┌──────────────┤
                                                     ▼              ▼
                                              [IRTLoss + XBM]  [Evaluation]
                                                     │              │
                                                     ▼              ▼
                                              [Optimizer]     [Metrics JSON]
                                                     │
                                                     ▼
                                              [Checkpoint .pth]
                                                     │
                                                     ▼
                                              [Demo / Inference]
```

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| DL Framework | PyTorch ≥ 2.0 |
| Vision Models | timm ≥ 0.9 (DeiT-Small) |
| Image Processing | Pillow, torchvision |
| Numerical | NumPy |
| Visualization | Matplotlib |
| Environment | Google Colab (T4 GPU) |
