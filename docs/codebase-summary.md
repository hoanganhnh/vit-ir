# Codebase Summary

## Project Stats

| Metric | Value |
|--------|-------|
| Total source LOC | ~3,750 |
| Core modules | 4 (`models`, `losses`, `data`, `utils`) |
| Scripts | 15 files (pipeline, training, evaluation) |
| Entry points | 2 (`train.py`, `demo.py`) |
| Language | Python 3.10+ |
| Dependencies | 7 (torch, torchvision, timm, Pillow, numpy, requests, tqdm) |

## Directory Structure

```
vit-ir/
├── src/                            # Core library (663 LOC)
│   ├── models/
│   │   └── backbone.py             # IRTModel + GeM pooling (214 LOC)
│   ├── losses/
│   │   └── irt_losses.py           # Contrastive + KoLeo + XBM (245 LOC)
│   ├── data/
│   │   └── datasets.py             # DataLoaders + CrossDomainDataset (245 LOC)
│   └── utils/
│       └── evaluation.py           # Recall@K, mAP, cross-domain eval (204 LOC)
├── scripts/                        # Dataset & training pipeline (2,237 LOC)
│   ├── 01_download_emnist.py       # EMNIST Letters download (121 LOC)
│   ├── 02_generate_rendered_fonts.py # Font image generation (259 LOC)
│   ├── 03_scrape_nasa_landsat.py   # NASA satellite scraper (296 LOC)
│   ├── 03b_download_nasa_colab.py  # Colab-specific NASA download (184 LOC)
│   ├── 03c_download_google_earth.py # Google Earth download (441 LOC)
│   ├── 04_augment_satellite.py     # Satellite augmentation (198 LOC)
│   ├── 05_verify_dataset.py        # Dataset verification (130 LOC)
│   ├── colab_setup.py              # Colab environment setup (323 LOC)
│   ├── colab_download_nasa.py      # NASA download helper (98 LOC)
│   ├── evaluate_cross_domain.py    # Cross-domain eval script (106 LOC)
│   ├── run_pipeline.py             # Master pipeline runner (81 LOC)
│   ├── train_stage1.sh             # Stage 1 training script
│   └── train_stage2.sh             # Stage 2 training script
├── train.py                        # Main training entry point (389 LOC)
├── demo.py                         # Interactive retrieval demo (219 LOC)
├── dataset/                        # Generated datasets (gitignored)
│   ├── emnist_letters/             # Tier 1: 145K images
│   ├── rendered_fonts/             # Tier 2: 15.6K images
│   ├── sat_fonts/                  # Tier 2b: 6.2K images
│   └── satellite_letters/          # Tier 3: 994 augmented images
├── checkpoints/                    # Trained model weights (gitignored)
├── notebook/                       # Jupyter notebooks (Colab)
│   ├── vit_ir_ptit_v1.ipynb
│   └── vit_ir_ptit_v2.ipynb
├── paper/                          # Reference paper
│   ├── Training Vision Transformers for Image Retrieval.pdf
│   └── paper.txt
├── docs/                           # Project documentation
│   ├── plans/                      # Design documents
│   └── ai/implementation/          # AI-generated implementation docs
├── requirements.txt                # Python dependencies
└── README.md                       # Project README
```

## Module Descriptions

### `src/models/backbone.py` — IRTModel

| Class/Function | Mô tả |
|---------------|-------|
| `GeM` | Generalized Mean Pooling với learnable exponent |
| `IRTModel` | Vision Transformer wrapper: backbone + pooling + projection + L2-norm |
| `IRTModel.freeze_layers()` | Freeze N transformer blocks cho fine-tuning |
| `build_model()` | Factory function tạo IRTModel từ config |

### `src/losses/irt_losses.py` — Loss Functions

| Class | Mô tả |
|-------|-------|
| `ContrastiveLoss` | Margin-based contrastive loss với XBM support |
| `KoLeoLoss` | Differential entropy regularization (nearest-neighbor) |
| `IRTLoss` | Combined loss: Contrastive + λ × KoLeo |
| `CrossBatchMemory` | FIFO queue lưu embeddings từ previous batches |

### `src/data/datasets.py` — Data Loading

| Class/Function | Mô tả |
|---------------|-------|
| `get_train_transform()` | Training augmentation pipeline |
| `get_eval_transform()` | Evaluation transform (resize + normalize) |
| `get_folder_dataset()` | ImageFolder dataset loader |
| `get_dataloader()` | DataLoader với system-aware defaults |
| `CrossDomainDataset` | Merge multiple datasets + oversampling |
| `get_cross_domain_loaders()` | Cross-domain train/test loaders |
| `get_emnist_loaders()` | EMNIST convenience loader |
| `get_rendered_fonts_loaders()` | Rendered fonts convenience loader |
| `get_satellite_loaders()` | Satellite convenience loader |

### `src/utils/evaluation.py` — Metrics

| Function | Mô tả |
|----------|-------|
| `extract_all_features()` | Extract features + labels cho entire dataset |
| `compute_recall_at_k()` | Recall@K metric (cosine similarity) |
| `compute_map()` | Mean Average Precision |
| `evaluate_retrieval()` | Full evaluation pipeline (same-domain) |
| `evaluate_cross_domain()` | Cross-domain evaluation (query ≠ gallery) |
| `print_metrics()` | Pretty-print metrics |

## Entry Points

### `train.py` — Training

```bash
# Stage 1: Shape pretraining
python train.py --dataset emnist --epochs 20

# Stage 2: Cross-domain fine-tuning
python train.py --dataset cross_domain --pretrained_from checkpoints/emnist_best.pth \
    --freeze_layers 6 --lr 1e-5 --xbm_size 2048 --epochs 30
```

### `demo.py` — Inference

```bash
python demo.py --checkpoint checkpoints/satellite_best.pth \
    --query path/to/query.jpg --gallery dataset/satellite_letters
```

## Dependency Graph

```
train.py ──▶ src/models/backbone.py (build_model)
         ──▶ src/losses/irt_losses.py (IRTLoss, CrossBatchMemory)
         ──▶ src/data/datasets.py (get_dataloaders)
         ──▶ src/utils/evaluation.py (evaluate_retrieval)

demo.py  ──▶ src/models/backbone.py (build_model)
         ──▶ src/data/datasets.py (transforms, loaders)
         ──▶ src/utils/evaluation.py (extract_all_features)

scripts/evaluate_cross_domain.py
         ──▶ src/models/backbone.py
         ──▶ src/data/datasets.py
         ──▶ src/utils/evaluation.py
```
