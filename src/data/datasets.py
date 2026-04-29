"""
Dataset loaders for SatLetter project
======================================
Provides PyTorch dataloaders for:
  - EMNIST Letters (proxy pretraining)
  - Rendered Fonts (bridge domain)
  - Satellite Letters (target domain)
  - CUB-200-2011 (baseline reproduction)
"""

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


# ── Standard transforms ─────────────────────────────────────────────────────

def get_train_transform(image_size: int = 224):
    """Training transforms following the IRT paper."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_eval_transform(image_size: int = 224):
    """Evaluation transforms (center crop)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ── Folder-based datasets ───────────────────────────────────────────────────

def get_folder_dataset(
    root: str,
    split: str = "train",
    image_size: int = 224,
) -> datasets.ImageFolder:
    """Load a dataset from folder structure: root/split/class_name/*.png"""
    split_dir = os.path.join(root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Dataset directory not found: {split_dir}")

    transform = get_train_transform(image_size) if split == "train" else get_eval_transform(image_size)
    dataset = datasets.ImageFolder(root=split_dir, transform=transform)

    print(f"Loaded {split} dataset from {split_dir}:")
    print(f"  Samples: {len(dataset)}")
    print(f"  Classes: {len(dataset.classes)}")

    return dataset


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Create a DataLoader with standard settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


# ── Convenience loaders ─────────────────────────────────────────────────────

def get_emnist_loaders(
    root: str = "dataset/emnist_letters",
    batch_size: int = 64,
    image_size: int = 224,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Get EMNIST Letters train and test dataloaders."""
    train_ds = get_folder_dataset(root, "train", image_size)
    test_ds = get_folder_dataset(root, "test", image_size)
    train_loader = get_dataloader(train_ds, batch_size, shuffle=True, num_workers=num_workers)
    test_loader = get_dataloader(test_ds, batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return train_loader, test_loader


def get_rendered_fonts_loaders(
    root: str = "dataset/rendered_fonts",
    batch_size: int = 64,
    image_size: int = 224,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Get Rendered Fonts train and test dataloaders."""
    train_ds = get_folder_dataset(root, "train", image_size)
    test_ds = get_folder_dataset(root, "test", image_size)
    train_loader = get_dataloader(train_ds, batch_size, shuffle=True, num_workers=num_workers)
    test_loader = get_dataloader(test_ds, batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return train_loader, test_loader


def get_satellite_loaders(
    root: str = "dataset/satellite_letters",
    batch_size: int = 64,
    image_size: int = 224,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Get Satellite Letters train and test dataloaders."""
    train_ds = get_folder_dataset(root, "train", image_size)
    test_ds = get_folder_dataset(root, "test", image_size)
    train_loader = get_dataloader(train_ds, batch_size, shuffle=True, num_workers=num_workers)
    test_loader = get_dataloader(test_ds, batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return train_loader, test_loader
