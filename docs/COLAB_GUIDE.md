# Hướng dẫn chạy SatLetter trên Google Colab

## Cách 1: Nhanh nhất — Clone + Run

Mở [Google Colab](https://colab.research.google.com/) → New Notebook → Copy từng cell dưới đây:

---

### Cell 1: Setup

```python
# Cài dependencies (Colab đã có torch, torchvision, numpy, PIL)
!pip install timm tqdm requests -q

# Clone repo (đã push lên GitHub)
!git clone https://github.com/hoanganhnh/vit-ir.git
%cd vit-ir

import torch
print(f"✅ PyTorch {torch.__version__}")
print(f"✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
```

### Cell 2: Download NASA + Generate Datasets

```python
# Download 72 ảnh NASA satellite letters (~1 phút)
!python scripts/03b_download_nasa_colab.py

# Generate sat_fonts: fonts overlay lên NASA textures (~3 phút)
!python scripts/02_generate_rendered_fonts.py 200 40 --sat

# Augment NASA cho evaluation gallery
!python scripts/04_augment_satellite.py 14

# Download EMNIST cho pretrain (~5 phút)
!python scripts/01_download_emnist.py 2000 400

# Verify
!python scripts/05_verify_dataset.py
```

### Cell 3: Stage 2 — Pretrain trên EMNIST

```python
# Pretrain: dạy model nhận dạng hình dạng chữ cái
# ~30-60 phút trên T4 GPU
!python train.py \
    --dataset emnist \
    --epochs 20 \
    --batch_size 64 \
    --lr 3e-5 \
    --lambda_koleo 0.7 \
    --eval_every 5 \
    --save_every 5
```

### Cell 4: Stage 3 — Fine-tune trên Satellite Fonts

```python
# Fine-tune: transfer EMNIST weights → train trên sat_fonts
# ~10-20 phút trên T4 GPU
!python train.py \
    --dataset sat_fonts \
    --epochs 15 \
    --batch_size 64 \
    --lr 1e-5 \
    --lambda_koleo 0.7 \
    --pretrained_from checkpoints/emnist_best.pth \
    --eval_every 3 \
    --save_every 5
```

### Cell 5: Cross-Domain Evaluation

```python
# Đánh giá: sat_fonts (synthetic) → NASA (real) retrieval
!python scripts/evaluate_cross_domain.py \
    --checkpoint checkpoints/sat_fonts_best.pth \
    --query_dir dataset/sat_fonts \
    --gallery_dir dataset/satellite_letters
```

### Cell 6: Demo Retrieval

```python
# Demo: input 1 ảnh → top-10 kết quả từ NASA gallery
!python demo.py \
    --checkpoint checkpoints/sat_fonts_best.pth \
    --query dataset/sat_fonts/test/S/satfont_S_00001.png \
    --gallery dataset/satellite_letters \
    --save retrieval_demo.png

from IPython.display import Image, display
display(Image('retrieval_demo.png'))
```

---

## Cách 2: Dùng Google Drive (lưu giữ data giữa sessions)

Thêm vào Cell 1:

```python
import os
from google.colab import drive

drive.mount('/content/drive', force_remount=True)

drive_path = '/content/drive/MyDrive'
# !cp -r /content/vit-ir {drive_path}

# os.chdir(drive_path)
print(f"📁 Thư mục làm việc hiện tại: {os.getcwd()}")
```

---

## Thời gian ước tính (T4 GPU)

| Step | Thời gian |
|------|-----------|
| Setup + Clone | ~1 phút |
| Download NASA + Generate sat_fonts | ~5 phút |
| Download EMNIST | ~5-10 phút |
| Stage 2: EMNIST pretrain (20 epochs) | ~30-60 phút |
| Stage 3: sat_fonts fine-tune (15 epochs) | ~10-20 phút |
| Evaluation + Demo | ~2-5 phút |
| **Tổng** | **~1-1.5 giờ** |

## Lưu ý

- Colab T4 có 15GB VRAM → đủ cho DeiT-Small (22M params), batch_size=64
- Nếu hết session, data mất → dùng Google Drive mount (Cách 2)
- File `scripts/colab_setup.py` chứa tất cả cells dạng Python strings để reference
