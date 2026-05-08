"""
Script 01: Download and prepare EMNIST Letters dataset
=====================================================
Downloads EMNIST Letters split and organizes into folder structure:
  dataset/emnist_letters/train/A/ ... Z/
  dataset/emnist_letters/test/A/ ... Z/
"""

import os
import sys
import string
from pathlib import Path

import torch
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm


def export_emnist_to_folders(
    root_dir: str = "dataset/emnist_letters",
    download_dir: str = "dataset/_raw_emnist",
    max_per_class_train: int = 2000,
    max_per_class_test: int = 400,
):
    """Download EMNIST Letters and export to folder-per-class structure."""

    print("=" * 60)
    print("STEP 1: Downloading EMNIST Letters dataset...")
    print("=" * 60)

    # Download EMNIST Letters split
    train_dataset = datasets.EMNIST(
        root=download_dir,
        split="letters",
        train=True,
        download=True,
    )
    test_dataset = datasets.EMNIST(
        root=download_dir,
        split="letters",
        train=False,
        download=True,
    )

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    # EMNIST Letters: labels 1-26 map to A-Z
    letters = list(string.ascii_uppercase)  # A-Z

    def export_split(dataset, split_name, max_per_class):
        """Export a dataset split to folders."""
        print(f"\n{'=' * 60}")
        print(f"STEP 2: Exporting {split_name} split...")
        print(f"{'=' * 60}")

        # Create directories
        for letter in letters:
            os.makedirs(os.path.join(root_dir, split_name, letter), exist_ok=True)

        # Track count per class
        class_counts = {letter: 0 for letter in letters}

        for idx in tqdm(range(len(dataset)), desc=f"Exporting {split_name}"):
            img, label = dataset[idx]

            # EMNIST labels: 1=A, 2=B, ..., 26=Z
            if label < 1 or label > 26:
                continue

            letter = letters[label - 1]

            if class_counts[letter] >= max_per_class:
                continue

            # EMNIST images need to be transposed (they're rotated/flipped)
            if isinstance(img, Image.Image):
                img = img.transpose(Image.TRANSPOSE)
            else:
                # If tensor
                img = transforms.ToPILImage()(img)
                img = img.transpose(Image.TRANSPOSE)

            # Convert to RGB and resize to 224x224
            img = img.convert("RGB")
            img = img.resize((224, 224), Image.BILINEAR)

            # Save
            filename = f"{letter}_{class_counts[letter]:05d}.png"
            filepath = os.path.join(root_dir, split_name, letter, filename)
            img.save(filepath)

            class_counts[letter] += 1

        # Summary
        print(f"\n{split_name} export summary:")
        total = 0
        for letter in letters:
            count = class_counts[letter]
            total += count
            print(f"  {letter}: {count} images")
        print(f"  TOTAL: {total} images")

    export_split(train_dataset, "train", max_per_class_train)
    export_split(test_dataset, "test", max_per_class_test)

    print(f"\n{'=' * 60}")
    print(f"DONE! EMNIST Letters exported to: {root_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    # Allow overriding max per class from command line
    max_train = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    max_test = int(sys.argv[2]) if len(sys.argv) > 2 else 400

    export_emnist_to_folders(
        max_per_class_train=max_train,
        max_per_class_test=max_test,
    )
