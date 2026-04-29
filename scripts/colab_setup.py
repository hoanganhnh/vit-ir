"""
SatLetter Dataset Collection - Google Colab Setup
==================================================
Copy and paste each cell into Google Colab to run the pipeline.
"""

# ===== CELL 1: Setup =====
CELL_1 = """
# @title 1. Setup - Install Dependencies & Clone Repo
!pip install torch torchvision Pillow numpy requests tqdm -q

# Clone your repo (update URL)
# !git clone https://github.com/YOUR_USERNAME/vit-ir.git
# %cd vit-ir

# Or upload scripts manually - they should be in scripts/ folder
import os
os.makedirs("scripts", exist_ok=True)
os.makedirs("dataset", exist_ok=True)
print("✅ Setup complete!")
"""

# ===== CELL 2: Download EMNIST =====
CELL_2 = """
# @title 2. Download EMNIST Letters Dataset (~145K images)
# This downloads the EMNIST Letters split and organizes into folders
# Takes ~5-10 minutes

!python scripts/01_download_emnist.py 2000 400
"""

# ===== CELL 3: Generate Rendered Fonts =====
CELL_3 = """
# @title 3. Generate Rendered Font Images (~15K images)
# Creates letter images with various fonts and satellite-like backgrounds
# Takes ~3-5 minutes

!python scripts/02_generate_rendered_fonts.py 500 100
"""

# ===== CELL 4: NASA Scraping =====
CELL_4 = """
# @title 4. Scrape NASA Landsat Alphabet (automated)
# Attempts to download satellite letter images from NASA
# Also generates a manual collection guide

!python scripts/03_scrape_nasa_landsat.py
"""

# ===== CELL 5: Manual Collection Guide =====
CELL_5 = """
# @title 5. View Manual Collection Guide
# Read this guide for collecting satellite letters from Google Earth

with open("dataset/satellite_letters/raw/nasa/COLLECTION_GUIDE.md", "r") as f:
    from IPython.display import Markdown, display
    display(Markdown(f.read()))
"""

# ===== CELL 6: Upload Manual Images =====
CELL_6 = """
# @title 6. Upload Manual Satellite Images
# Upload your manually collected satellite letter images
# They should be organized by letter: A/, B/, ..., Z/

from google.colab import files
import shutil

# Option A: Upload a zip file
print("Upload a ZIP file containing folders A/ through Z/ with satellite images")
uploaded = files.upload()

for filename in uploaded:
    if filename.endswith('.zip'):
        !unzip -o {filename} -d dataset/satellite_letters/raw/google_earth/
        print(f"✅ Extracted {filename}")
    else:
        print(f"⚠️ Skipping {filename} (not a zip file)")
"""

# ===== CELL 7: Augment =====
CELL_7 = """
# @title 7. Augment Satellite Images (after manual collection)
# Applies augmentation and splits into train/test

!python scripts/04_augment_satellite.py 5
"""

# ===== CELL 8: Verify =====
CELL_8 = """
# @title 8. Verify All Datasets
!python scripts/05_verify_dataset.py
"""

# ===== CELL 9: Preview =====
CELL_9 = """
# @title 9. Preview Dataset Samples
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

# Preview each dataset
for name, path in [
    ("EMNIST Letters", "dataset/emnist_letters"),
    ("Rendered Fonts", "dataset/rendered_fonts"),
    ("Satellite Letters", "dataset/satellite_letters"),
]:
    if os.path.isdir(os.path.join(path, "train")):
        preview_dataset(path, name)
    else:
        print(f"⚠️ {name} not found at {path}")
"""

if __name__ == "__main__":
    print("=" * 60)
    print("SatLetter - Colab Notebook Cells")
    print("=" * 60)
    print("\nCopy each CELL variable into separate Colab cells.")
    print("Run them in order (1 → 9).")
    print("\nCells:")
    for i in range(1, 10):
        cell = globals()[f"CELL_{i}"]
        title = cell.strip().split("\n")[0]
        print(f"  Cell {i}: {title}")
