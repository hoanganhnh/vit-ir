"""
Script 04: Augment satellite letter images
==========================================
Takes raw satellite letter images and applies augmentations
to increase dataset size. Then splits into train/test.
Output: dataset/satellite_letters/train|test/A/ ... Z/
"""

import os
import sys
import string
import random
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import numpy as np
from tqdm import tqdm


def augment_image(img: Image.Image, size: int = 224) -> list[Image.Image]:
    """Apply various augmentations to an image. Returns list of augmented images."""
    augmented = []

    # Ensure RGB and correct size
    img = img.convert("RGB").resize((size, size), Image.BICUBIC)

    # Original (resized)
    augmented.append(img.copy())

    # Horizontal flip
    augmented.append(ImageOps.mirror(img))

    # Vertical flip
    augmented.append(ImageOps.flip(img))

    # Rotations
    for angle in [90, 180, 270]:
        augmented.append(img.rotate(angle, resample=Image.BICUBIC, expand=False))

    # Small random rotations
    for _ in range(2):
        angle = random.uniform(-30, 30)
        augmented.append(img.rotate(angle, resample=Image.BICUBIC, expand=False))

    # Random crops (zoom in)
    for _ in range(2):
        w, h = img.size
        crop_ratio = random.uniform(0.7, 0.9)
        cw, ch = int(w * crop_ratio), int(h * crop_ratio)
        x = random.randint(0, w - cw)
        y = random.randint(0, h - ch)
        cropped = img.crop((x, y, x + cw, y + ch))
        augmented.append(cropped.resize((size, size), Image.BICUBIC))

    # Color jitter
    for _ in range(2):
        aug = img.copy()
        if random.random() < 0.5:
            aug = ImageEnhance.Brightness(aug).enhance(random.uniform(0.7, 1.3))
        if random.random() < 0.5:
            aug = ImageEnhance.Contrast(aug).enhance(random.uniform(0.7, 1.3))
        if random.random() < 0.5:
            aug = ImageEnhance.Color(aug).enhance(random.uniform(0.7, 1.3))
        augmented.append(aug)

    # Gaussian blur
    augmented.append(img.filter(ImageFilter.GaussianBlur(radius=1.5)))

    # Sharpen
    augmented.append(img.filter(ImageFilter.SHARPEN))

    return augmented


def process_satellite_dataset(
    raw_dirs: list[str] = None,
    output_dir: str = "dataset/satellite_letters",
    target_size: int = 224,
    test_ratio: float = 0.2,
    augmentation_multiplier: int = 5,
):
    """
    Process raw satellite images:
    1. Collect all raw images per letter
    2. Apply augmentation to reach target count
    3. Split into train/test
    """

    if raw_dirs is None:
        raw_dirs = [
            "dataset/satellite_letters/raw/nasa",
            "dataset/satellite_letters/raw/google_earth",
            "dataset/satellite_letters/raw/sentinel",
        ]

    print("=" * 60)
    print("Processing Satellite Letter Dataset")
    print("=" * 60)

    letters = list(string.ascii_uppercase)

    # Collect all raw images per letter
    print("\n--- Collecting raw images ---")
    raw_images = {letter: [] for letter in letters}

    for raw_dir in raw_dirs:
        if not os.path.isdir(raw_dir):
            print(f"  Skipping {raw_dir} (not found)")
            continue

        for letter in letters:
            letter_dir = os.path.join(raw_dir, letter)
            if not os.path.isdir(letter_dir):
                continue

            for fname in os.listdir(letter_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".bmp")):
                    raw_images[letter].append(os.path.join(letter_dir, fname))

    # Summary of raw images
    total_raw = 0
    for letter in letters:
        count = len(raw_images[letter])
        total_raw += count
        status = "✓" if count > 0 else "✗ EMPTY"
        print(f"  {letter}: {count} raw images {status}")
    print(f"  TOTAL RAW: {total_raw}")

    if total_raw == 0:
        print("\nNo raw satellite images found!")
        print("Please collect images first using the COLLECTION_GUIDE.md")
        print("Place them in: dataset/satellite_letters/raw/google_earth/{letter}/")
        return

    # Create train/test directories
    for split in ["train", "test"]:
        for letter in letters:
            os.makedirs(os.path.join(output_dir, split, letter), exist_ok=True)

    # Process each letter
    print(f"\n--- Augmenting (x{augmentation_multiplier}) and splitting ---")

    for letter in tqdm(letters, desc="Processing"):
        images = raw_images[letter]
        if not images:
            continue

        # Load and augment all images
        all_augmented = []
        for img_path in images:
            try:
                img = Image.open(img_path)
                augmented_list = augment_image(img, target_size)
                # Take up to augmentation_multiplier images per original
                all_augmented.extend(augmented_list[:augmentation_multiplier])
            except Exception as e:
                print(f"  Error processing {img_path}: {e}")

        if not all_augmented:
            continue

        # Shuffle and split
        random.shuffle(all_augmented)
        n_test = max(1, int(len(all_augmented) * test_ratio))
        n_train = len(all_augmented) - n_test

        test_images = all_augmented[:n_test]
        train_images = all_augmented[n_test:]

        # Save train
        for i, img in enumerate(train_images):
            img.save(os.path.join(output_dir, "train", letter, f"sat_{letter}_{i:05d}.png"))

        # Save test
        for i, img in enumerate(test_images):
            img.save(os.path.join(output_dir, "test", letter, f"sat_{letter}_{i:05d}.png"))

    # Final summary
    print(f"\n{'=' * 60}")
    print("FINAL DATASET SUMMARY")
    print(f"{'=' * 60}")

    total_train, total_test = 0, 0
    for letter in letters:
        train_dir = os.path.join(output_dir, "train", letter)
        test_dir = os.path.join(output_dir, "test", letter)
        n_train = len(os.listdir(train_dir)) if os.path.isdir(train_dir) else 0
        n_test = len(os.listdir(test_dir)) if os.path.isdir(test_dir) else 0
        total_train += n_train
        total_test += n_test
        print(f"  {letter}: train={n_train}, test={n_test}")

    print(f"\n  TOTAL: train={total_train}, test={total_test}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    aug_mult = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    process_satellite_dataset(augmentation_multiplier=aug_mult)
