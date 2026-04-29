"""
IRT Training Script
====================
Main training loop for Image Retrieval with Transformers.
Supports training on EMNIST, Rendered Fonts, Satellite Letters, and CUB-200.

Usage:
    python train.py --dataset emnist --epochs 20
    python train.py --dataset satellite --pretrained_from checkpoints/emnist_best.pth
    python train.py --dataset rendered_fonts --lambda_koleo 0.7
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.backbone import build_model
from src.losses.irt_losses import IRTLoss, CrossBatchMemory
from src.data.datasets import (
    get_emnist_loaders,
    get_rendered_fonts_loaders,
    get_satellite_loaders,
    get_folder_dataset,
    get_dataloader,
)
from src.utils.evaluation import evaluate_retrieval, print_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="IRT Training for SatLetter")

    # Dataset
    parser.add_argument("--dataset", type=str, default="emnist",
                        choices=["emnist", "rendered_fonts", "satellite", "cub200"],
                        help="Dataset to train on")
    parser.add_argument("--data_root", type=str, default="dataset",
                        help="Root directory for datasets")

    # Model
    parser.add_argument("--backbone", type=str, default="deit_small_patch16_224",
                        help="Backbone model name (timm)")
    parser.add_argument("--pooling", type=str, default="cls",
                        choices=["cls", "avg", "max", "gem"],
                        help="Feature pooling method")
    parser.add_argument("--embed_dim", type=int, default=None,
                        help="Projection dimensionality (None = use backbone dim)")

    # Loss
    parser.add_argument("--margin", type=float, default=0.5,
                        help="Contrastive loss margin β")
    parser.add_argument("--lambda_koleo", type=float, default=0.7,
                        help="KoLeo regularization strength λ")
    parser.add_argument("--no_koleo", action="store_true",
                        help="Disable KoLeo (IRTL instead of IRTR)")

    # Cross-Batch Memory
    parser.add_argument("--xbm_size", type=int, default=0,
                        help="Cross-batch memory size (0 = disabled)")

    # Training
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Input image size")

    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--pretrained_from", type=str, default=None,
                        help="Load pretrained IRT weights from checkpoint")
    parser.add_argument("--eval_every", type=int, default=5,
                        help="Evaluate every N epochs")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save checkpoint every N epochs")

    # Misc
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto/cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def get_dataloaders(args) -> tuple[DataLoader, DataLoader]:
    """Get train and test dataloaders based on args."""
    dataset_paths = {
        "emnist": os.path.join(args.data_root, "emnist_letters"),
        "rendered_fonts": os.path.join(args.data_root, "rendered_fonts"),
        "satellite": os.path.join(args.data_root, "satellite_letters"),
        "cub200": os.path.join(args.data_root, "cub200"),
    }

    root = dataset_paths[args.dataset]

    train_ds = get_folder_dataset(root, "train", args.image_size)
    test_ds = get_folder_dataset(root, "test", args.image_size)

    train_loader = get_dataloader(
        train_ds, args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    test_loader = get_dataloader(
        test_ds, args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False,
    )

    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: IRTLoss,
    optimizer: optim.Optimizer,
    memory: CrossBatchMemory | None,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_contrastive = 0.0
    total_koleo = 0.0
    n_batches = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward
        embeddings = model(images)  # (B, D)

        # Get memory
        mem_emb, mem_lab = None, None
        if memory is not None:
            mem_data = memory.get()
            if mem_data is not None:
                mem_emb, mem_lab = mem_data

        # Compute loss
        losses = criterion(embeddings, labels, mem_emb, mem_lab)

        # Backward
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        # Update memory
        if memory is not None:
            memory.enqueue(embeddings.detach(), labels.detach())

        # Accumulate stats
        total_loss += losses["total"].item()
        total_contrastive += losses["contrastive"].item()
        total_koleo += losses["koleo"].item()
        n_batches += 1

        # Log progress
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
            avg_loss = total_loss / n_batches
            print(
                f"  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                f"Loss: {avg_loss:.4f} "
                f"(contr: {total_contrastive/n_batches:.4f}, "
                f"koleo: {total_koleo/n_batches:.4f})"
            )

    return {
        "loss": total_loss / n_batches,
        "loss_contrastive": total_contrastive / n_batches,
        "loss_koleo": total_koleo / n_batches,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: dict,
    args: argparse.Namespace,
    filename: str,
):
    """Save training checkpoint."""
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    path = os.path.join(args.checkpoint_dir, filename)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "args": vars(args),
    }, path)

    print(f"  Checkpoint saved: {path}")


def main():
    args = parse_args()

    # Setup
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    print(f"\n{'=' * 60}")
    print(f"IRT Training — SatLetter Project")
    print(f"{'=' * 60}")
    print(f"Device:    {device}")
    print(f"Dataset:   {args.dataset}")
    print(f"Backbone:  {args.backbone}")
    print(f"Pooling:   {args.pooling}")
    print(f"Loss:      Contrastive(β={args.margin}) + {'λ=' + str(args.lambda_koleo) + ' KoLeo' if not args.no_koleo else 'No KoLeo'}")
    print(f"XBM size:  {args.xbm_size}")
    print(f"Epochs:    {args.epochs}")
    print(f"Batch:     {args.batch_size}")
    print(f"LR:        {args.lr}")

    # Data
    print(f"\n--- Loading Data ---")
    train_loader, test_loader = get_dataloaders(args)

    # Model
    print(f"\n--- Building Model ---")
    model = build_model(
        backbone=args.backbone,
        pooling=args.pooling,
        embed_dim=args.embed_dim,
        pretrained=True,
    )

    # Load pretrained IRT weights if specified
    if args.pretrained_from:
        print(f"Loading pretrained weights from: {args.pretrained_from}")
        ckpt = torch.load(args.pretrained_from, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    model = model.to(device)

    # Loss
    criterion = IRTLoss(
        margin=args.margin,
        lambda_koleo=args.lambda_koleo,
        use_koleo=not args.no_koleo,
    )

    # Cross-Batch Memory
    memory = None
    if args.xbm_size > 0:
        memory = CrossBatchMemory(args.xbm_size, model.output_dim).to(device)
        print(f"XBM initialized: size={args.xbm_size}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Training loop
    print(f"\n{'=' * 60}")
    print(f"Starting Training")
    print(f"{'=' * 60}")

    best_recall = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, memory, device, epoch
        )

        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s")

        # Evaluate
        eval_metrics = {}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            print(f"  Evaluating...")
            eval_metrics = evaluate_retrieval(model, test_loader, device=str(device))
            print_metrics(eval_metrics, prefix=f"  Epoch {epoch}")

            # Save best
            r1 = eval_metrics.get("R@1", 0)
            if r1 > best_recall:
                best_recall = r1
                save_checkpoint(model, optimizer, epoch, eval_metrics, args,
                                f"{args.dataset}_best.pth")

        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, eval_metrics or train_metrics, args,
                            f"{args.dataset}_epoch{epoch}.pth")

        # Record history
        history.append({
            "epoch": epoch,
            "time": elapsed,
            **train_metrics,
            **eval_metrics,
        })

    # Final evaluation
    print(f"\n{'=' * 60}")
    print(f"Training Complete!")
    print(f"{'=' * 60}")
    final_metrics = evaluate_retrieval(model, test_loader, device=str(device))
    print_metrics(final_metrics, prefix="Final")
    print(f"Best R@1: {best_recall:.2f}%")

    # Save final checkpoint
    save_checkpoint(model, optimizer, args.epochs, final_metrics, args,
                    f"{args.dataset}_final.pth")

    # Save training history
    history_path = os.path.join(args.checkpoint_dir, f"{args.dataset}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved: {history_path}")


if __name__ == "__main__":
    main()
