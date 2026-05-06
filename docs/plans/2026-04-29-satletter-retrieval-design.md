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

## 3. Dataset Strategy — 3 Tiers (Approach B: NASA-Only)

### Tier 1: EMNIST Letters (Proxy Pretraining)

- **Nguồn**: torchvision.datasets.EMNIST(split='letters')
- **Kích thước**: ~145,000 train / ~14,800 test (26 classes)
- **Mục đích**: Dạy model nhận biết hình dạng chữ cái tổng quát
- **Xử lý**: Resize 28→224, convert grayscale→RGB

### Tier 2: Synthetic Satellite Fonts (Bridge Domain — Approach B)

- **Nguồn**: Rendered fonts overlay lên random crops từ ảnh NASA satellite thật
- **Kích thước mục tiêu**: ~6,240 ảnh (200 train + 40 test per class × 26)
- **Mục đích**: Bridge domain gap tự nhiên — chữ cái trên nền satellite texture thật
- **Kỹ thuật**: Alpha compositing, semi-transparent blending, contrast color selection
- **Biến thể**: rotation ±35°, Gaussian blur, brightness/contrast jitter
- **Script**: `scripts/02b_generate_sat_fonts.py`
- **Ưu điểm**: Không cần manual collection, tự động generate, domain adaptation tự nhiên

### Tier 3: NASA SatLetter (Evaluation Only)

- **Nguồn**: NASA "Your Name in Landsat" gallery (71 ảnh, ~2-5/class)
- **Kích thước**: 71 raw → ~994 augmented (x14)
- **Augmentation**: Rotation, flip, crop, color jitter → x14
- **Mục đích**: **Evaluation gallery only** — dùng để đánh giá cross-domain retrieval
- **Lý do**: 71 ảnh quá ít để train (overfitting), nhưng đủ để làm ground truth test

### Dataset Format

```
dataset/
├── emnist_letters/           # Tier 1: EMNIST pretrain
│   ├── train/
│   │   ├── A/ ... Z/
│   └── test/
│       ├── A/ ... Z/
├── sat_fonts/                # Tier 2: Synthetic satellite fonts (TRAINING)
│   ├── train/                #   200/class × 26 = 5,200 ảnh
│   │   ├── A/ ... Z/
│   └── test/                 #   40/class × 26 = 1,040 ảnh
│       ├── A/ ... Z/
├── rendered_fonts/           # (backup) Pure rendered fonts
│   ├── train/
│   │   ├── A/ ... Z/
│   └── test/
│       ├── A/ ... Z/
└── satellite_letters/        # Tier 3: NASA thật (EVALUATION ONLY)
    ├── raw/
    │   └── nasa/             #   71 ảnh gốc
    ├── train/                #   Augmented for gallery
    │   ├── A/ ... Z/
    └── test/                 #   Augmented for evaluation
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

### Stage 3: Fine-tune trên Synthetic Satellite Fonts

- Transfer weights từ Stage 2
- Fine-tune trên `sat_fonts` dataset (5,200 ảnh synthetic)
- **Freeze backbone**, chỉ train projection head (tránh overfit)
- Learning rate thấp: 1e-5
- Evaluate bằng cross-domain retrieval: sat_fonts query → NASA gallery
- So sánh: ImageNet pretrain vs EMNIST pretrain

## 5. Evaluation

### Metrics

- **Recall@K** (K=1, 5, 10) — metric chính
- **mAP** — nếu có đủ dữ liệu

### Cross-Domain Evaluation (Approach B)

| Evaluation Mode | Query | Gallery | Mục đích |
|----------------|-------|---------|----------|
| **Cross-domain** | sat_fonts test | NASA augmented | Đo khả năng generalize synthetic→real |
| **Self-retrieval** | NASA augmented | NASA augmented | Baseline trên real data |
| **In-domain** | sat_fonts test | sat_fonts test | Sanity check |

- Script: `scripts/evaluate_cross_domain.py`

### Ablation Studies

| Experiment      | Mô tả                                 |
| --------------- | ------------------------------------- |
| Pretrain source | ImageNet vs EMNIST vs Fonts+EMNIST    |
| Loss function   | Contrastive only vs Contrastive+KoLeo |
| Pooling method  | CLS vs Avg vs GeM                     |
| Descriptor dim  | 384 vs 128 (PCA)                      |
| Backbone freeze | Freeze vs Full fine-tune              |

### Qualitative Demo

- Input 1 ảnh → Hiển thị grid top-10 kết quả
- Visualization t-SNE/UMAP embedding space theo chữ cái

## 6. Constraints

- **GPU**: Google Colab T4 (15GB VRAM)
- **Timeline**: 4 tuần
- **Storage**: Colab ~100GB

## 7. Timeline

| Tuần | Nhiệm vụ                                                       |
| ---- | -------------------------------------------------------------- |
| 1    | NASA scrape (71 ảnh) + EMNIST + generate sat_fonts (tự động)   |
| 2    | Reproduce IRT trên CUB-200, bắt đầu train EMNIST               |
| 3    | Fine-tune sat_fonts, cross-domain evaluation                    |
| 4    | Ablation studies, demo, viết báo cáo                           |

## 8. Deliverables

1. Codebase Python (PyTorch) chạy trên Colab
2. SatFonts synthetic dataset (~6,240 ảnh) + NASA evaluation set (71 raw + augmented)
3. Trained models + cross-domain evaluation results
4. Báo cáo đồ án
5. Demo notebook (input ảnh → retrieval results)
