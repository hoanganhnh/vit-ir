# Approach B (NASA-Only) Implementation Plan

> **For Antigravity:** REQUIRED WORKFLOW: Use `.agent/workflows/execute-plan.md` to execute this plan in single-flow mode.

**Goal:** Sửa script `02_generate_rendered_fonts.py` để overlay fonts lên real NASA satellite textures, tạo dataset `sat_fonts` cho training. NASA images thật chỉ dùng để evaluation.

**Architecture:** Thay hàm `generate_satellite_background()` (tạo noise giả) bằng random crops từ 71 ảnh NASA thật. Thêm alpha blending cho letter → nền satellite trông tự nhiên hơn. Thêm `sat_fonts` vào train.py + cross-domain eval script.

**Tech Stack:** Python, Pillow (PIL), PyTorch, existing IRT pipeline

---

### Task 1: Sửa `02_generate_rendered_fonts.py` — thêm real NASA texture

**Files:**

- Modify: `scripts/02_generate_rendered_fonts.py`

**Step 1: Thêm NASA texture loader**

Thêm sau hàm `find_available_fonts()` (line 47):

```python
def load_nasa_textures(
    nasa_dir: str = "dataset/satellite_letters/raw/nasa",
    min_size: int = 224,
) -> list[Image.Image]:
    """Load all NASA satellite images as texture sources."""
    from pathlib import Path
    textures = []
    if not os.path.isdir(nasa_dir):
        return textures
    for letter_dir in sorted(Path(nasa_dir).iterdir()):
        if not letter_dir.is_dir():
            continue
        for img_path in letter_dir.iterdir():
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                if min(img.size) >= min_size:
                    textures.append(img)
                else:
                    textures.append(img.resize((min_size, min_size), Image.BICUBIC))
            except Exception:
                pass
    return textures


def random_nasa_crop(textures: list[Image.Image], size: int = 224) -> Image.Image:
    """Random crop from a random NASA texture with augmentation."""
    img = random.choice(textures)
    w, h = img.size
    if w < size or h < size:
        crop = img.resize((size, size), Image.BICUBIC)
    else:
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        crop = img.crop((x, y, x + size, y + size))
    # Random augment texture
    if random.random() < 0.3:
        from PIL import ImageOps
        crop = ImageOps.mirror(crop)
    if random.random() < 0.3:
        from PIL import ImageOps
        crop = ImageOps.flip(crop)
    if random.random() < 0.2:
        crop = crop.rotate(random.choice([90, 180, 270]))
    if random.random() < 0.3:
        crop = ImageEnhance.Color(crop).enhance(random.uniform(0.6, 1.4))
    return crop
```

**Step 2: Sửa `render_letter()` — dùng NASA texture + alpha blending**

Thay background selection (line 77-80) và letter rendering (line 89-96) để dùng NASA texture + alpha compositing:

```python
def render_letter(letter, font_path, size=224, nasa_textures=None):
    """Render a single letter with random augmentations.
    
    If nasa_textures provided, uses real satellite imagery as background
    with alpha-blended letter for natural appearance.
    """
    # Background selection
    if nasa_textures and random.random() < 0.85:
        img = random_nasa_crop(nasa_textures, size)
    elif random.random() < 0.7:
        img = generate_satellite_background(size)
    else:
        img = Image.new("RGB", (size, size), tuple(random.randint(20, 230) for _ in range(3)))

    font_size = random.randint(int(size * 0.35), int(size * 0.85))
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Use alpha compositing for natural blending when using NASA textures
    if nasa_textures:
        letter_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(letter_layer)
        
        # Pick color that contrasts with background
        bg_arr = np.array(img)
        bg_mean = bg_arr.mean(axis=(0, 1))
        offset = random.randint(40, 120)
        color = tuple(
            int(np.clip(bg_mean[i] + random.choice([-1, 1]) * offset, 0, 255))
            for i in range(3)
        )
        alpha = random.randint(100, 220)
        
        bbox = draw.textbbox((0, 0), letter, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ox = random.randint(-int(size * 0.1), int(size * 0.1))
        oy = random.randint(-int(size * 0.1), int(size * 0.1))
        x = (size - tw) // 2 + ox - bbox[0]
        y = (size - th) // 2 + oy - bbox[1]
        draw.text((x, y), letter, fill=(*color, alpha), font=font)
        
        # Blur letter for natural look
        letter_layer = letter_layer.filter(
            ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.5))
        )
        img = Image.alpha_composite(img.convert("RGBA"), letter_layer).convert("RGB")
    else:
        draw = ImageDraw.Draw(img)
        color = tuple(random.randint(0, 255) for _ in range(3))
        bbox = draw.textbbox((0, 0), letter, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ox = random.randint(-int(size * 0.1), int(size * 0.1))
        oy = random.randint(-int(size * 0.1), int(size * 0.1))
        x = (size - tw) // 2 + ox - bbox[0]
        y = (size - th) // 2 + oy - bbox[1]
        draw.text((x, y), letter, fill=color, font=font)

    img = img.rotate(random.uniform(-35, 35), resample=Image.BICUBIC)
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    if random.random() < 0.4:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.3:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    return img
```

**Step 3: Sửa `generate_rendered_fonts_dataset()` — thêm NASA mode + sat_fonts output**

Thêm parameter `nasa_dir` và `use_nasa_textures`, thay đổi output dir mặc định:

```python
def generate_rendered_fonts_dataset(
    output_dir="dataset/rendered_fonts",
    num_train=500,
    num_test=100,
    image_size=224,
    nasa_dir="dataset/satellite_letters/raw/nasa",
    use_nasa_textures=False,
):
    """Generate the complete rendered fonts dataset.
    
    When use_nasa_textures=True, overlays fonts on real NASA satellite textures
    and saves to dataset/sat_fonts/ (Approach B).
    """
    if use_nasa_textures:
        output_dir = "dataset/sat_fonts"

    print("=" * 60)
    title = "Generating Satellite Font Dataset (Approach B)" if use_nasa_textures else "Generating Rendered Fonts Dataset"
    print(title)
    print("=" * 60)

    # Load NASA textures if requested
    nasa_textures = None
    if use_nasa_textures:
        nasa_textures = load_nasa_textures(nasa_dir, image_size)
        if not nasa_textures:
            print("ERROR: No NASA textures found. Run 03b_download_nasa_colab.py first.")
            sys.exit(1)
        print(f"Loaded {len(nasa_textures)} NASA satellite textures")

    fonts = find_available_fonts()
    print(f"Found {len(fonts)} fonts")
    letters = list(string.ascii_uppercase)

    for split, count in [("train", num_train), ("test", num_test)]:
        print(f"\n--- Generating {split} ({count}/class) ---")
        for letter in tqdm(letters, desc=split):
            d = os.path.join(output_dir, split, letter)
            os.makedirs(d, exist_ok=True)
            for i in range(count):
                ch = letter if random.random() < 0.7 else letter.lower()
                img = render_letter(ch, random.choice(fonts), image_size, nasa_textures)
                prefix = "satfont" if use_nasa_textures else letter
                img.save(os.path.join(d, f"{prefix}_{letter}_{i:05d}.png"))
        print(f"{split}: {count * len(letters)} images")

    print(f"\nDONE! Saved to: {output_dir}")
```

**Step 4: Sửa `__main__` — thêm `--sat` flag**

```python
if __name__ == "__main__":
    import argparse as _ap
    p = _ap.ArgumentParser(description="Generate rendered font letter images")
    p.add_argument("num_train", nargs="?", type=int, default=500)
    p.add_argument("num_test", nargs="?", type=int, default=100)
    p.add_argument("--sat", action="store_true",
                   help="Use NASA satellite textures (Approach B → dataset/sat_fonts/)")
    p.add_argument("--nasa-dir", default="dataset/satellite_letters/raw/nasa",
                   help="Path to NASA raw images")
    a = p.parse_args()
    generate_rendered_fonts_dataset(
        num_train=a.num_train,
        num_test=a.num_test,
        use_nasa_textures=a.sat,
        nasa_dir=a.nasa_dir,
    )
```

**Step 5: Verify script runs**

Run: `python scripts/02_generate_rendered_fonts.py 10 5 --sat`
Expected: Creates `dataset/sat_fonts/train/` (260 images) and `dataset/sat_fonts/test/` (130 images) using NASA textures

**Step 6: Commit**

Run: `git add scripts/02_generate_rendered_fonts.py && git commit -m "feat: add NASA satellite texture mode to rendered fonts generator (Approach B)"`

---

### Task 2: Thêm `sat_fonts` vào `train.py`

**Files:**

- Modify: `train.py:44-46` (dataset choices)
- Modify: `train.py:112-117` (dataset paths)

**Step 1: Thêm `sat_fonts` vào argument choices**

Line 44-46, thêm `"sat_fonts"`:

```python
    parser.add_argument("--dataset", type=str, default="emnist",
                        choices=["emnist", "rendered_fonts", "satellite", "sat_fonts", "cub200"],
                        help="Dataset to train on")
```

**Step 2: Thêm `sat_fonts` vào dataset_paths**

Line 112-117, thêm entry:

```python
    dataset_paths = {
        "emnist": os.path.join(args.data_root, "emnist_letters"),
        "rendered_fonts": os.path.join(args.data_root, "rendered_fonts"),
        "satellite": os.path.join(args.data_root, "satellite_letters"),
        "sat_fonts": os.path.join(args.data_root, "sat_fonts"),
        "cub200": os.path.join(args.data_root, "cub200"),
    }
```

**Step 3: Commit**

Run: `git add train.py && git commit -m "feat: add sat_fonts dataset option to training script"`

---

### Task 3: Tạo cross-domain evaluation script

**Files:**

- Create: `scripts/evaluate_cross_domain.py`

**Step 1: Tạo script**

```python
"""
Cross-domain evaluation for Approach B
========================================
Query: synthetic sat_fonts test images
Gallery: real NASA satellite images (augmented)

Usage:
    python scripts/evaluate_cross_domain.py \
        --checkpoint checkpoints/sat_fonts_best.pth \
        --query_dir dataset/sat_fonts \
        --gallery_dir dataset/satellite_letters
"""

import os
import sys
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.backbone import build_model
from src.data.datasets import get_folder_dataset, get_dataloader
from src.utils.evaluation import (
    extract_all_features,
    compute_recall_at_k,
    compute_map,
    print_metrics,
)


def main():
    parser = argparse.ArgumentParser(description="Cross-domain retrieval evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--query_dir", type=str, default="dataset/sat_fonts")
    parser.add_argument("--gallery_dir", type=str, default="dataset/satellite_letters")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    ckpt = torch.load(args.checkpoint, map_location="cpu")
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
    print(f"Loaded model from: {args.checkpoint}")

    # Query set (synthetic sat_fonts test)
    print(f"\n--- Query: {args.query_dir}/test ---")
    query_ds = get_folder_dataset(args.query_dir, "test", args.image_size)
    query_loader = get_dataloader(query_ds, args.batch_size, shuffle=False, drop_last=False)

    # Gallery set (real NASA satellite test)
    print(f"\n--- Gallery: {args.gallery_dir}/test ---")
    gallery_ds = get_folder_dataset(args.gallery_dir, "test", args.image_size)
    gallery_loader = get_dataloader(gallery_ds, args.batch_size, shuffle=False, drop_last=False)

    # Extract features
    print("\nExtracting features...")
    q_feats, q_labels = extract_all_features(model, query_loader, device)
    g_feats, g_labels = extract_all_features(model, gallery_loader, device)

    # Cross-domain: sat_fonts → NASA
    print(f"\n{'=' * 60}")
    print("CROSS-DOMAIN: SatFonts → NASA Satellite")
    print(f"{'=' * 60}")
    print(f"Queries: {q_feats.size(0)} | Gallery: {g_feats.size(0)}")

    recall = compute_recall_at_k(
        q_feats, q_labels, g_feats, g_labels,
        k_values=[1, 5, 10], exclude_self=False,
    )
    recall["mAP"] = compute_map(
        q_feats, q_labels, g_feats, g_labels, exclude_self=False,
    )
    print_metrics(recall, prefix="Cross-domain")

    # Self-retrieval: NASA → NASA
    print(f"\n{'=' * 60}")
    print("SELF-RETRIEVAL: NASA → NASA")
    print(f"{'=' * 60}")

    self_recall = compute_recall_at_k(
        g_feats, g_labels, g_feats, g_labels,
        k_values=[1, 5, 10], exclude_self=True,
    )
    self_recall["mAP"] = compute_map(
        g_feats, g_labels, g_feats, g_labels, exclude_self=True,
    )
    print_metrics(self_recall, prefix="NASA self")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

Run: `git add scripts/evaluate_cross_domain.py && git commit -m "feat: add cross-domain evaluation script (Approach B)"`

---

### Task 4: Update pipeline runner

**Files:**

- Modify: `scripts/run_pipeline.py:49-54`

**Step 1: Thêm step 2b cho sat_fonts**

Sửa `steps` dict và default run order:

```python
    steps = {
        1: ("scripts/01_download_emnist.py", [str(args.emnist_max_train), str(args.emnist_max_test)]),
        2: ("scripts/02_generate_rendered_fonts.py", [str(args.fonts_train), str(args.fonts_test)]),
        3: ("scripts/03_scrape_nasa_landsat.py", []),
        4: ("scripts/04_augment_satellite.py", [str(args.aug_multiplier)]),
        5: ("scripts/02_generate_rendered_fonts.py", [str(args.fonts_train), str(args.fonts_test), "--sat"]),
        6: ("scripts/05_verify_dataset.py", []),
    }
```

Sửa default run order (line 65):

```python
        # Default: 1→EMNIST, 2→fonts, 3→NASA download, 4→augment NASA, 5→sat_fonts, 6→verify
        steps_to_run = [1, 2, 3, 4, 5, 6] if args.all else [1, 3, 4, 5, 6]
```

**Step 2: Commit**

Run: `git add scripts/run_pipeline.py && git commit -m "feat: add sat_fonts generation to pipeline runner"`

---

### Task 5: End-to-end verification

**Step 1: Download NASA images (nếu chưa có)**

Run: `python scripts/03b_download_nasa_colab.py`
Expected: 71 images in `dataset/satellite_letters/raw/nasa/`

**Step 2: Generate sat_fonts**

Run: `python scripts/02_generate_rendered_fonts.py 200 40 --sat`
Expected: `dataset/sat_fonts/train/` = 5,200 ảnh, `dataset/sat_fonts/test/` = 1,040 ảnh

**Step 3: Augment NASA cho evaluation**

Run: `python scripts/04_augment_satellite.py 14`
Expected: `dataset/satellite_letters/train/` và `test/` populated

**Step 4: Verify datasets**

Run: `python scripts/05_verify_dataset.py`
Expected: Tất cả datasets có correct counts

**Step 5: Test training (dry run, 1 epoch, CPU)**

Run: `python train.py --dataset sat_fonts --epochs 1 --batch_size 16 --device cpu --eval_every 1`
Expected: 1 epoch completes, prints loss + metrics

**Step 6: Final commit**

Run: `git add -A && git commit -m "feat: complete Approach B (NASA-only) pipeline"`
