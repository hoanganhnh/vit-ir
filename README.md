# SatLetter: Satellite Letter-Shape Image Retrieval

Ứng dụng Vision Transformers (IRT) cho truy xuất ảnh vệ tinh chứa hình dạng chữ cái.

**Based on**: [Training Vision Transformers for Image Retrieval](paper/) (El-Nouby et al., ICML 2022)

## Mô tả

Hệ thống **image-to-image retrieval** cho ảnh vệ tinh:
- **Input**: Ảnh chữ cái (rendered font / ảnh vệ tinh mẫu)
- **Output**: Top-K ảnh vệ tinh có hình dạng tương tự từ cơ sở dữ liệu

## Cấu trúc dự án

```
vit-ir/
├── paper/                          # Bài báo gốc
├── docs/plans/                     # Design documents
├── scripts/                        # Dataset collection pipeline
│   ├── 01_download_emnist.py       # Download EMNIST Letters
│   ├── 02_generate_rendered_fonts.py # Generate font images
│   ├── 03_scrape_nasa_landsat.py   # Scrape NASA satellite letters
│   ├── 04_augment_satellite.py     # Augment satellite images
│   ├── 05_verify_dataset.py        # Verify dataset stats
│   └── run_pipeline.py             # Master pipeline runner
├── dataset/                        # Dataset (generated)
│   ├── emnist_letters/             # Tier 1: Proxy pretraining
│   ├── rendered_fonts/             # Tier 2: Bridge domain
│   └── satellite_letters/          # Tier 3: Target domain
├── requirements.txt
└── README.md
```

## Quick Start (Google Colab)

```bash
# 1. Install dependencies
pip install torch torchvision Pillow numpy requests tqdm

# 2. Run full dataset pipeline (Steps 1-3 + verification)
python scripts/run_pipeline.py

# 3. After manual satellite collection, run augmentation
python scripts/run_pipeline.py --step 4

# 4. Verify final dataset
python scripts/run_pipeline.py --step 5
```

## Dataset Pipeline

| Step | Script | Description | Auto? |
|------|--------|-------------|-------|
| 1 | `01_download_emnist.py` | Download EMNIST Letters (145K images) | ✅ |
| 2 | `02_generate_rendered_fonts.py` | Generate rendered fonts (15.6K images) | ✅ |
| 3 | `03_scrape_nasa_landsat.py` | Scrape NASA + generate collection guide | ✅ |
| 4 | `04_augment_satellite.py` | Augment raw satellite images | ⚠️ After manual collection |
| 5 | `05_verify_dataset.py` | Verify all datasets | ✅ |

## Timeline

| Tuần | Nhiệm vụ |
|------|----------|
| 1 | Thu thập dataset (pipeline + manual Google Earth) |
| 2 | Reproduce IRT trên CUB-200, train EMNIST |
| 3 | Fine-tune SatLetter, evaluation |
| 4 | Ablation, demo, báo cáo |
