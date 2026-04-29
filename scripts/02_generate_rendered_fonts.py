"""
Script 02: Generate rendered font letter images
================================================
Creates letter images from various fonts with augmentations and
optional satellite-like texture backgrounds.
Output: dataset/rendered_fonts/train|test/A/ ... Z/
"""

import os
import sys
import string
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
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


def render_letter(letter, font_path, size=224):
    """Render a single letter with random augmentations."""
    if random.random() < 0.7:
        img = generate_satellite_background(size)
    else:
        img = Image.new("RGB", (size, size), tuple(random.randint(20, 230) for _ in range(3)))

    draw = ImageDraw.Draw(img)
    font_size = random.randint(int(size * 0.4), int(size * 0.85))
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except (OSError, IOError):
        font = ImageFont.load_default()

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
    return img


def generate_rendered_fonts_dataset(
    output_dir="dataset/rendered_fonts",
    num_train=500,
    num_test=100,
    image_size=224,
):
    """Generate the complete rendered fonts dataset."""
    print("=" * 60)
    print("Generating Rendered Fonts Dataset")
    print("=" * 60)

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
                img = render_letter(ch, random.choice(fonts), image_size)
                img.save(os.path.join(d, f"{letter}_{i:05d}.png"))
        print(f"{split}: {count * len(letters)} images")

    print(f"\nDONE! Saved to: {output_dir}")


if __name__ == "__main__":
    nt = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    ne = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    generate_rendered_fonts_dataset(num_train=nt, num_test=ne)
