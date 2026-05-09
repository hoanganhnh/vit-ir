"""
Script 02: Generate rendered font letter images
================================================
Creates letter images from various fonts with augmentations and
optional satellite-like texture backgrounds.

Modes:
  - Default: dataset/rendered_fonts/train|test/A/ ... Z/
  - --sat:   dataset/sat_fonts/train|test/A/ ... Z/  (NASA textures)
"""

import os
import sys
import string
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps
import numpy as np
from tqdm import tqdm


SYSTEM_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
]


def find_available_fonts():
    """Find all available .ttf/.otf fonts on the system."""
    available = [p for p in SYSTEM_FONT_PATHS if os.path.exists(p)]
    for font_dir in ["/usr/share/fonts", "/usr/local/share/fonts"]:
        if os.path.isdir(font_dir):
            for root, _, files in os.walk(font_dir):
                for f in files:
                    if f.endswith((".ttf", ".otf")):
                        path = os.path.join(root, f)
                        if path not in available:
                            available.append(path)
    if not available:
        print("WARNING: No system fonts found, using PIL default.")
        available = [None]
    return available


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
        crop = ImageOps.mirror(crop)
    if random.random() < 0.3:
        crop = ImageOps.flip(crop)
    if random.random() < 0.2:
        crop = crop.rotate(random.choice([90, 180, 270]))
    if random.random() < 0.3:
        crop = ImageEnhance.Color(crop).enhance(random.uniform(0.6, 1.4))
    return crop


def generate_satellite_background(size=224):
    """Generate pseudo satellite-like texture background."""
    bg = random.choice(["terrain", "water", "urban", "field", "plain"])
    if bg == "terrain":
        arr = np.random.randint(60, 140, (size, size, 3), dtype=np.uint8)
        arr[:, :, 1] = np.clip(arr[:, :, 1].astype(int) + 20, 0, 255).astype(np.uint8)
    elif bg == "water":
        arr = np.random.randint(20, 80, (size, size, 3), dtype=np.uint8)
        arr[:, :, 2] = np.clip(arr[:, :, 2].astype(int) + 60, 0, 255).astype(np.uint8)
    elif bg == "urban":
        base = np.random.randint(100, 180, (size, size), dtype=np.uint8)
        arr = np.stack([base, base, base], axis=-1)
    elif bg == "field":
        arr = np.random.randint(80, 160, (size, size, 3), dtype=np.uint8)
        arr[:, :, 1] = np.clip(arr[:, :, 1].astype(int) + 40, 0, 255).astype(np.uint8)
    else:
        c = [random.randint(60, 200) for _ in range(3)]
        arr = np.full((size, size, 3), c, dtype=np.uint8)
    noise = np.random.randint(-15, 15, (size, size, 3), dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
    return img


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
