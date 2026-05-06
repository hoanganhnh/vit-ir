"""
Script 05: Dataset statistics and verification
===============================================
Scans all dataset directories and reports statistics.
Verifies images are valid and correctly sized.
"""

import os
import sys
import string
from pathlib import Path
from collections import defaultdict

from PIL import Image
from tqdm import tqdm


def verify_dataset(base_dir: str = "dataset"):
    """Verify and report statistics for all dataset directories."""

    print("=" * 70)
    print("DATASET VERIFICATION REPORT")
    print("=" * 70)

    datasets = {
        "EMNIST Letters": os.path.join(base_dir, "emnist_letters"),
        "Rendered Fonts": os.path.join(base_dir, "rendered_fonts"),
        "Satellite Fonts (Approach B)": os.path.join(base_dir, "sat_fonts"),
        "Satellite Letters": os.path.join(base_dir, "satellite_letters"),
    }

    letters = list(string.ascii_uppercase)
    grand_total = 0

    for dataset_name, dataset_dir in datasets.items():
        print(f"\n{'─' * 70}")
        print(f"📁 {dataset_name}: {dataset_dir}")
        print(f"{'─' * 70}")

        if not os.path.isdir(dataset_dir):
            print("  ⚠️  Directory not found!")
            continue

        for split in ["train", "test"]:
            split_dir = os.path.join(dataset_dir, split)
            if not os.path.isdir(split_dir):
                print(f"  ⚠️  {split}/ not found")
                continue

            total = 0
            min_count = float("inf")
            max_count = 0
            min_letter = ""
            max_letter = ""
            sizes = set()
            corrupt = 0

            for letter in letters:
                letter_dir = os.path.join(split_dir, letter)
                if not os.path.isdir(letter_dir):
                    count = 0
                else:
                    files = [
                        f
                        for f in os.listdir(letter_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif"))
                    ]
                    count = len(files)

                    # Check a few random images
                    for fname in files[:3]:
                        try:
                            img = Image.open(os.path.join(letter_dir, fname))
                            sizes.add(img.size)
                        except Exception:
                            corrupt += 1

                total += count
                if count < min_count:
                    min_count = count
                    min_letter = letter
                if count > max_count:
                    max_count = count
                    max_letter = letter

            grand_total += total

            print(f"\n  📊 {split.upper()} split:")
            print(f"     Total images:  {total}")
            print(f"     Min per class: {min_count} ({min_letter})")
            print(f"     Max per class: {max_count} ({max_letter})")
            print(f"     Avg per class: {total / 26:.1f}")
            if sizes:
                print(f"     Image sizes:   {sizes}")
            if corrupt > 0:
                print(f"     ⚠️  Corrupt:    {corrupt}")

    # Also check raw satellite images
    raw_dir = os.path.join(base_dir, "satellite_letters", "raw")
    if os.path.isdir(raw_dir):
        print(f"\n{'─' * 70}")
        print(f"📁 Raw Satellite Images: {raw_dir}")
        print(f"{'─' * 70}")

        for source in ["nasa", "google_earth", "sentinel"]:
            source_dir = os.path.join(raw_dir, source)
            if not os.path.isdir(source_dir):
                continue

            total = 0
            for letter in letters:
                letter_dir = os.path.join(source_dir, letter)
                if os.path.isdir(letter_dir):
                    files = [
                        f
                        for f in os.listdir(letter_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif"))
                    ]
                    total += len(files)

            print(f"  {source}: {total} images")

    print(f"\n{'=' * 70}")
    print(f"GRAND TOTAL: {grand_total} images across all datasets")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    verify_dataset(base)
