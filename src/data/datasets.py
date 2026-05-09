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


# ── Cross-domain dataset ────────────────────────────────────────────────────


class CrossDomainDataset(Dataset):
    """Merge multiple ImageFolder datasets with unified labels and oversampling.

    All datasets must share the same class names (A-Z).
    Smaller datasets can be oversampled to balance representation in batches.

    Args:
        source_datasets: List of ImageFolder datasets to merge
        oversample_factors: Dict mapping dataset index to repeat factor.
                           e.g. {2: 10} repeats source_datasets[2] samples 10x.
    """

    def __init__(
        self,
        source_datasets: list[datasets.ImageFolder],
        oversample_factors: dict[int, int] | None = None,
    ):
        super().__init__()
        self.source_datasets = source_datasets
        self.oversample_factors = oversample_factors or {}

        # Build unified class mapping from first dataset
        self.classes = source_datasets[0].classes
        self.class_to_idx = source_datasets[0].class_to_idx

        # Build unified sample list: (dataset_idx, sample_idx, unified_label)
        self.samples = []
        for ds_idx, ds in enumerate(source_datasets):
            # Map dataset's class indices to unified indices
            ds_class_map = {}
            for cls_name, cls_idx in ds.class_to_idx.items():
                if cls_name in self.class_to_idx:
                    ds_class_map[cls_idx] = self.class_to_idx[cls_name]

            repeat = self.oversample_factors.get(ds_idx, 1)
            for _ in range(repeat):
                for sample_idx in range(len(ds)):
                    _, original_label = ds.samples[sample_idx]
                    unified_label = ds_class_map.get(original_label, original_label)
                    self.samples.append((ds_idx, sample_idx, unified_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ds_idx, sample_idx, unified_label = self.samples[idx]
        image, _ = self.source_datasets[ds_idx][sample_idx]
        return image, unified_label


def get_cross_domain_loaders(
    data_root: str = "dataset",
    batch_size: int = 64,
    image_size: int = 224,
    satellite_oversample: int = 10,
    num_workers: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Get cross-domain train/test loaders merging EMNIST + Fonts + Satellite.

    Satellite images are oversampled to balance representation.
    All datasets must have train/ and test/ splits with A-Z class folders.
    """
    emnist_train = get_folder_dataset(os.path.join(data_root, "emnist_letters"), "train", image_size)
    fonts_train = get_folder_dataset(os.path.join(data_root, "rendered_fonts"), "train", image_size)
    sat_train = get_folder_dataset(os.path.join(data_root, "satellite_letters"), "train", image_size)

    train_ds = CrossDomainDataset(
        source_datasets=[emnist_train, fonts_train, sat_train],
        oversample_factors={2: satellite_oversample},
    )

    # Test: just satellite for cross-domain eval gallery
    sat_test = get_folder_dataset(os.path.join(data_root, "satellite_letters"), "test", image_size)

    print(f"CrossDomain train: {len(train_ds)} samples (sat oversample={satellite_oversample}x)")

    train_loader = get_dataloader(train_ds, batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = get_dataloader(sat_test, batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    return train_loader, test_loader


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
