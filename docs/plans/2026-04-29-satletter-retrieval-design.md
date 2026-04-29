# SatLetter Retrieval System — Design Document

**Date**: 2026-04-29
**Based on**: "Training Vision Transformers for Image Retrieval" (El-Nouby et al., ICML 2022)
**Goal**: Đồ án môn học — Ứng dụng IRT cho truy xuất ảnh vệ tinh chứa hình dạng chữ cái

---

## 1. Problem Statement

Xây dựng hệ thống **image retrieval** cho ảnh vệ tinh chứa hình dạng giống chữ cái:
- **Input**: Một ảnh (chữ cái rendered / ảnh vệ tinh mẫu)
- **Output**: Top-K ảnh vệ tinh tương tự nhất từ cơ sở dữ liệu

Ví dụ: Input ảnh chữ "S" → Trả về ảnh vệ tinh con sông uốn cong hình chữ S.

## 2. Method — IRT (Image Retrieval with Transformers)

Áp dụng trực tiếp pipeline từ paper gốc:

### 2.1 Backbone
- **DeiT-Small** (22M params, 8.4 GFLOPS) — phù hợp Colab T4
- Pretrained trên ImageNet1k
- Output: CLS token 384-D làm image descriptor
- L2-normalize → cosine similarity cho retrieval

### 2.2 Loss Function
```
L = L_contrastive + λ * L_KoLeo

L_contrastive: Contrastive loss với margin β, cross-batch memory
L_KoLeo: Differential entropy regularization (chống feature collapse)
```

### 2.3 Hyperparameters (theo paper)
- Optimizer: AdamW, lr=3e-5, weight_decay=5e-4
- Batch size: 64
- Contrastive margin β = 0.5
- Entropy regularization λ = 0.7
- Image size: 224×224

## 3. Dataset Strategy — 3 Tiers

### Tier 1: EMNIST Letters (Proxy Pretraining)
- **Nguồn**: torchvision.datasets.EMNIST(split='letters')
- **Kích thước**: ~145,000 train / ~14,800 test (26 classes)
- **Mục đích**: Dạy model nhận biết hình dạng chữ cái tổng quát
- **Xử lý**: Resize 28→224, convert grayscale→RGB

### Tier 2: Rendered Fonts (Bridge Domain)
- **Nguồn**: Tự generate từ ~50 Google Fonts phổ biến
- **Kích thước mục tiêu**: ~13,000 ảnh (500/class × 26)
- **Mục đích**: Bridge gap giữa handwriting (EMNIST) và structural shapes
- **Biến thể**: rotation ±30°, scale, thickness, satellite-like texture background

### Tier 3: SatLetter (Target Domain)
- **Nguồn**:
  1. NASA "Your Name in Landsat" gallery (scrape)
  2. Google Earth Pro (manual collection)
  3. Sentinel Hub API (keyword search by coordinates)
- **Kích thước mục tiêu**: 1,300-2,600 ảnh (50-100/class × 26)
- **Augmentation**: Rotation, flip, crop, color jitter → x3-5x
- **Mục đích**: Target domain chính

### Dataset Format
```
dataset/
├── emnist_letters/
│   ├── train/
│   │   ├── A/ ... Z/
│   └── test/
│       ├── A/ ... Z/
├── rendered_fonts/
│   ├── train/
│   │   ├── A/ ... Z/
│   └── test/
│       ├── A/ ... Z/
└── satellite_letters/
    ├── raw/
    │   ├── nasa/
    │   ├── google_earth/
    │   └── sentinel/
    ├── train/
    │   ├── A/ ... Z/
    └── test/
        ├── A/ ... Z/
```

## 4. Training Pipeline

### Stage 1: Reproduce Baseline (CUB-200)
- Chạy IRT trên CUB-200-2011 (11,788 ảnh, 200 classes)
- Xác nhận pipeline hoạt động đúng
- Target: Recall@1 ~ 74.7% (theo paper)

### Stage 2: Pretrain trên EMNIST + Fonts
- Train IRT (DeiT-S + contrastive + KoLeo) trên EMNIST Letters
- Optionally fine-tune thêm trên rendered fonts

### Stage 3: Fine-tune trên SatLetter
- Transfer weights từ Stage 2
- Fine-tune trên satellite letter dataset
- So sánh: ImageNet pretrain vs EMNIST pretrain

## 5. Evaluation

### Metrics
- **Recall@K** (K=1, 5, 10) — metric chính
- **mAP** — nếu có đủ dữ liệu

### Ablation Studies
| Experiment | Mô tả |
|---|---|
| Pretrain source | ImageNet vs EMNIST vs Fonts+EMNIST |
| Loss function | Contrastive only vs Contrastive+KoLeo |
| Pooling method | CLS vs Avg vs GeM |
| Descriptor dim | 384 vs 128 (PCA) |

### Qualitative Demo
- Input 1 ảnh → Hiển thị grid top-10 kết quả
- Visualization t-SNE/UMAP embedding space theo chữ cái

## 6. Constraints
- **GPU**: Google Colab T4 (15GB VRAM)
- **Timeline**: 4 tuần
- **Storage**: Colab ~100GB

## 7. Timeline
| Tuần | Nhiệm vụ |
|---|---|
| 1 | Thu thập dataset (NASA scrape + Google Earth + EMNIST + fonts) |
| 2 | Reproduce IRT trên CUB-200, bắt đầu train EMNIST |
| 3 | Fine-tune SatLetter, chạy evaluation |
| 4 | Ablation studies, demo, viết báo cáo |

## 8. Deliverables
1. Codebase Python (PyTorch) chạy trên Colab
2. SatLetter dataset (~1,300+ ảnh vệ tinh)
3. Trained models + evaluation results
4. Báo cáo đồ án
5. Demo notebook (input ảnh → retrieval results)
