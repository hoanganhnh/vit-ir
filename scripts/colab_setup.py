"""
SatLetter Training - Google Colab Notebook (Approach B)
========================================================
Copy each CELL into separate Colab cells and run in order.

Pipeline:
  1. Setup & Clone repo
  2. Download NASA images
  3. Generate sat_fonts (fonts on NASA textures)
  4. Augment NASA for evaluation
  5. Verify datasets
  6. Preview samples
  7. Train on sat_fonts
  8. Cross-domain evaluation
  9. Demo retrieval
"""

# ===== CELL 1: Setup & Clone =====
CELL_1 = """
# @title 1. Setup — Clone Repo & Install Dependencies
# Colab đã có sẵn: torch, torchvision, numpy, PIL, matplotlib
# Chỉ cần cài thêm timm (Vision Transformer backbone)

!pip install timm tqdm requests -q

# Clone repo
!git clone https://github.com/hoanganhnh/vit-ir.git
%cd vit-ir

import torch
print(f"✅ PyTorch {torch.__version__}")
print(f"✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
print(f"✅ CUDA: {torch.cuda.is_available()}")
"""

# ===== CELL 2: Download NASA =====
CELL_2 = """
# @title 2. Download NASA Landsat Alphabet (72 ảnh vệ tinh)
# Tải 72 ảnh chữ cái từ NASA "Your Name in Landsat"
# Mất ~1-2 phút

!python scripts/03b_download_nasa_colab.py
"""

# ===== CELL 3: Generate sat_fonts =====
CELL_3 = """
# @title 3. Generate Satellite Fonts — Approach B (fonts overlay lên NASA textures)
# Render chữ cái bằng fonts → overlay lên random crops từ ảnh NASA thật
# 200 train + 40 test per class × 26 = 6,240 ảnh
# Mất ~3-5 phút

!python scripts/02_generate_rendered_fonts.py 200 40 --sat
"""

# ===== CELL 4: Augment NASA =====
CELL_4 = """
# @title 4. Augment NASA cho Evaluation Gallery
# Augment 72 ảnh NASA thật × 14 biến thể → ~1,000 ảnh evaluation
# NASA images thật CHỈ DÙNG ĐỂ TEST (không train)

!python scripts/04_augment_satellite.py 14
"""

# ===== CELL 5: Download EMNIST =====
CELL_5 = """
# @title 5. Download EMNIST Letters (Pretrain dataset)
# ~145K handwritten letter images cho Stage 2 pretrain
# Mất ~5-10 phút

!python scripts/01_download_emnist.py 2000 400
"""

# ===== CELL 6: Verify =====
CELL_6 = """
# @title 6. Verify All Datasets
!python scripts/05_verify_dataset.py
"""

# ===== CELL 7: Preview =====
CELL_7 = """
# @title 7. Preview Dataset Samples
import os
import random
import matplotlib.pyplot as plt
from PIL import Image

def preview_dataset(dataset_dir, title, n_letters=6, n_per_letter=4):
    fig, axes = plt.subplots(n_letters, n_per_letter, figsize=(3*n_per_letter, 3*n_letters))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    letters = sorted(os.listdir(os.path.join(dataset_dir, "train")))
    letters = [l for l in letters if len(l) == 1 and l.isalpha()]
    sample_letters = random.sample(letters, min(n_letters, len(letters)))

    for i, letter in enumerate(sample_letters):
        letter_dir = os.path.join(dataset_dir, "train", letter)
        images = [f for f in os.listdir(letter_dir) if f.endswith(('.png', '.jpg'))]
        sample_imgs = random.sample(images, min(n_per_letter, len(images)))

        for j, img_name in enumerate(sample_imgs):
            img = Image.open(os.path.join(letter_dir, img_name))
            axes[i][j].imshow(img)
            axes[i][j].set_title(f"{letter}" if j == 0 else "")
            axes[i][j].axis('off')

    plt.tight_layout()
    plt.show()

# Preview sat_fonts (synthetic training data)
preview_dataset("dataset/sat_fonts", "Satellite Fonts (Training Data)")

# Preview real NASA satellite letters (evaluation data)
if os.path.isdir("dataset/satellite_letters/train"):
    preview_dataset("dataset/satellite_letters", "NASA Satellite Letters (Evaluation)")
"""

# ===== CELL 8: Train Stage 2 — EMNIST Pretrain =====
CELL_8 = """
# @title 8. Stage 2: Pretrain trên EMNIST Letters
# Dạy model nhận dạng hình dạng chữ cái tổng quát
# DeiT-Small + Contrastive + KoLeo Loss
# ~20 epochs, mất ~30-60 phút trên T4

!python train.py \\
    --dataset emnist \\
    --epochs 20 \\
    --batch_size 64 \\
    --lr 3e-5 \\
    --lambda_koleo 0.7 \\
    --eval_every 5 \\
    --save_every 5
"""

# ===== CELL 9: Train Stage 3 — Fine-tune sat_fonts =====
CELL_9 = """
# @title 9. Stage 3: Fine-tune trên Satellite Fonts (Approach B)
# Transfer weights từ EMNIST → fine-tune trên sat_fonts
# Learning rate thấp hơn, ít epochs hơn (tránh overfit)
# Mất ~10-20 phút trên T4

!python train.py \\
    --dataset sat_fonts \\
    --epochs 15 \\
    --batch_size 64 \\
    --lr 1e-5 \\
    --lambda_koleo 0.7 \\
    --pretrained_from checkpoints/emnist_best.pth \\
    --eval_every 3 \\
    --save_every 5
"""

# ===== CELL 10: Cross-Domain Evaluation =====
CELL_10 = """
# @title 10. Cross-Domain Evaluation: SatFonts → NASA
# Đánh giá: model train trên synthetic có generalize sang ảnh NASA thật không?
# Query = sat_fonts test | Gallery = NASA augmented test

!python scripts/evaluate_cross_domain.py \\
    --checkpoint checkpoints/sat_fonts_best.pth \\
    --query_dir dataset/sat_fonts \\
    --gallery_dir dataset/satellite_letters
"""

# ===== CELL 11: Demo Retrieval =====
CELL_11 = """
# @title 11. Demo: Image-to-Image Retrieval
# Chọn 1 ảnh query → tìm top-10 ảnh satellite tương tự nhất

import os
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image
import sys

sys.path.insert(0, '.')
from src.models.backbone import build_model
from src.data.datasets import get_eval_transform, get_folder_dataset, get_dataloader
from src.utils.evaluation import extract_all_features

# Load trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load("checkpoints/sat_fonts_best.pth", map_location="cpu")
model_args = ckpt.get("args", {})
model = build_model(
    backbone=model_args.get("backbone", "deit_small_patch16_224"),
    pooling=model_args.get("pooling", "cls"),
    embed_dim=model_args.get("embed_dim", None),
    pretrained=False,
)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(device)
model.eval()

# Build gallery from NASA test set
gallery_ds = get_folder_dataset("dataset/satellite_letters", "test", 224)
gallery_loader = get_dataloader(gallery_ds, 64, shuffle=False, drop_last=False)
gallery_feats, gallery_labels = extract_all_features(model, gallery_loader, device)
gallery_paths = [gallery_ds.samples[i][0] for i in range(len(gallery_ds))]
class_names = gallery_ds.classes

# Pick a random query from sat_fonts test
query_dir = "dataset/sat_fonts/test"
letters = sorted(os.listdir(query_dir))
letter = random.choice(letters)
letter_dir = os.path.join(query_dir, letter)
query_file = random.choice([f for f in os.listdir(letter_dir) if f.endswith('.png')])
query_path = os.path.join(letter_dir, query_file)

# Extract query feature
transform = get_eval_transform(224)
query_img = Image.open(query_path).convert("RGB")
query_tensor = transform(query_img).unsqueeze(0).to(device)
with torch.no_grad():
    query_feat = model(query_tensor).cpu()

# Find top-10 similar
similarities = torch.mm(query_feat, gallery_feats.t()).squeeze(0)
topk_sim, topk_idx = similarities.topk(10)

# Visualize
fig, axes = plt.subplots(1, 11, figsize=(33, 3.5))
axes[0].imshow(query_img)
axes[0].set_title(f"QUERY: {letter}", fontsize=12, fontweight="bold", color="red")
axes[0].axis("off")

for i, (sim, idx) in enumerate(zip(topk_sim.tolist(), topk_idx.tolist())):
    img = Image.open(gallery_paths[idx]).convert("RGB")
    label = class_names[gallery_labels[idx].item()]
    axes[i+1].imshow(img)
    match = "✓" if label == letter else "✗"
    color = "green" if label == letter else "gray"
    axes[i+1].set_title(f"#{i+1} {label} {match}\\n{sim:.3f}", fontsize=9, color=color)
    axes[i+1].axis("off")

plt.suptitle("SatLetter Retrieval: Synthetic Query → NASA Gallery", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("retrieval_demo.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\\n📁 Saved to retrieval_demo.png")
"""

# ===== CELL 12: t-SNE Visualization =====
CELL_12 = """
# @title 12. t-SNE Embedding Visualization
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys, os

sys.path.insert(0, '.')
from src.models.backbone import build_model
from src.data.datasets import get_folder_dataset, get_dataloader
from src.utils.evaluation import extract_all_features

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load("checkpoints/sat_fonts_best.pth", map_location="cpu")
model_args = ckpt.get("args", {})
model = build_model(
    backbone=model_args.get("backbone", "deit_small_patch16_224"),
    pooling=model_args.get("pooling", "cls"),
    embed_dim=model_args.get("embed_dim", None),
    pretrained=False,
)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(device)
model.eval()

# Extract features from NASA test set
ds = get_folder_dataset("dataset/satellite_letters", "test", 224)
loader = get_dataloader(ds, 64, shuffle=False, drop_last=False)
feats, labels = extract_all_features(model, loader, device)

# t-SNE
tsne = TSNE(n_components=2, perplexity=min(30, len(feats)-1), random_state=42)
coords = tsne.fit_transform(feats.numpy())

# Plot
plt.figure(figsize=(12, 10))
class_names = ds.classes
colors = plt.cm.tab20(np.linspace(0, 1, 26))

for i, letter in enumerate(class_names):
    mask = labels.numpy() == i
    if mask.sum() > 0:
        plt.scatter(coords[mask, 0], coords[mask, 1], c=[colors[i]], label=letter, s=30, alpha=0.7)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=8)
plt.title("t-SNE of NASA Satellite Letter Embeddings", fontsize=14)
plt.tight_layout()
plt.savefig("tsne_embeddings.png", dpi=150, bbox_inches="tight")
plt.show()
print("📁 Saved to tsne_embeddings.png")
"""


if __name__ == "__main__":
    print("=" * 60)
    print("SatLetter - Colab Notebook Cells (Approach B)")
    print("=" * 60)
    print()
    print("Mở Google Colab → New Notebook → Copy từng CELL vào:")
    print()
    cells = {
        1: "Setup & Clone repo",
        2: "Download NASA (72 ảnh)",
        3: "Generate sat_fonts (6,240 ảnh synthetic)",
        4: "Augment NASA cho evaluation (~1,000 ảnh)",
        5: "Download EMNIST (pretrain, ~145K ảnh)",
        6: "Verify datasets",
        7: "Preview samples",
        8: "Stage 2: Pretrain EMNIST (~30-60 min)",
        9: "Stage 3: Fine-tune sat_fonts (~10-20 min)",
        10: "Cross-domain evaluation",
        11: "Demo retrieval visualization",
        12: "t-SNE embedding visualization",
    }
    for i, desc in cells.items():
        print(f"  Cell {i:2d}: {desc}")
    print()
    print("⏱️  Tổng thời gian ước tính: ~1-2 giờ (T4 GPU)")
    print("💾 Tổng storage: ~500MB")
