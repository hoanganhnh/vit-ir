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
