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


# ── System-aware defaults ────────────────────────────────────────────────────

def _safe_num_workers() -> int:
    """Return a safe num_workers based on available CPU cores.
    
    Google Colab free tier typically has 2 vCPUs.
    Using more workers than CPUs causes context-switch thrashing and freezes.
    """
    try:
        cpu_count = os.cpu_count() or 1
        # Reserve 1 core for the main process; at least 0 (main-process loading)
        return max(0, min(cpu_count - 1, cpu_count))
    except Exception:
        return 0


def _safe_pin_memory() -> bool:
    """Only pin memory when a CUDA device is available."""
    return torch.cuda.is_available()


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
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    drop_last: bool = True,
) -> DataLoader:
    """Create a DataLoader with system-aware defaults.
    
    When num_workers or pin_memory are None, they are auto-detected:
      - num_workers: min(cpu_count - 1, cpu_count), at least 0
      - pin_memory: True only when CUDA is available
    """
    if num_workers is None:
        num_workers = _safe_num_workers()
    if pin_memory is None:
        pin_memory = _safe_pin_memory()

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
    num_workers: int | None = None,
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
    num_workers: int | None = None,
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
    num_workers: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Get Satellite Letters train and test dataloaders."""
    train_ds = get_folder_dataset(root, "train", image_size)
    test_ds = get_folder_dataset(root, "test", image_size)
    train_loader = get_dataloader(train_ds, batch_size, shuffle=True, num_workers=num_workers)
    test_loader = get_dataloader(test_ds, batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return train_loader, test_loader
