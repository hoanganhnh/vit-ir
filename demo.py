"""
SatLetter Retrieval Demo
=========================
Interactive demo for querying satellite letter images.
Input: an image → Output: top-K most similar satellite images.

Usage:
    python demo.py --checkpoint checkpoints/satellite_best.pth --query path/to/query.jpg
    python demo.py --checkpoint checkpoints/satellite_best.pth --gallery dataset/satellite_letters/test
"""

import os
import sys
import argparse

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.backbone import build_model
from src.data.datasets import get_eval_transform, get_folder_dataset, get_dataloader
from src.utils.evaluation import extract_all_features
from torchvision.datasets import ImageFolder


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load trained IRT model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
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

    print(f"Loaded model from: {checkpoint_path}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}")
    if "metrics" in ckpt:
        print(f"  Metrics: {ckpt['metrics']}")

    return model


def extract_query_feature(model, image_path: str, device: str = "cuda", image_size: int = 224):
    """Extract feature from a single query image."""
    transform = get_eval_transform(image_size)
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feature = model(img_tensor)  # (1, D)

    return feature.cpu()


def build_gallery_index(model, gallery_dir: str, device: str = "cuda", image_size: int = 224):
    """Build gallery index: extract features for all gallery images.

    Handles both split (train/test) and flat (class_name/) gallery structures.
    """
    # Try test split first
    test_dir = os.path.join(gallery_dir, "test")
    if os.path.isdir(test_dir):
        dataset = get_folder_dataset(gallery_dir, split="test", image_size=image_size)
    else:
        # Flat gallery: gallery_dir contains class folders directly
        dataset = ImageFolder(
            root=gallery_dir,
            transform=get_eval_transform(image_size),
        )
        print(f"Loaded flat gallery from {gallery_dir}:")
        print(f"  Samples: {len(dataset)}")
        print(f"  Classes: {len(dataset.classes)}")

    loader = get_dataloader(dataset, batch_size=64, shuffle=False, drop_last=False)
    features, labels = extract_all_features(model, loader, device)

    # Get image paths
    image_paths = [dataset.samples[i][0] for i in range(len(dataset))]
    class_names = dataset.classes

    return features, labels, image_paths, class_names


def retrieve_similar(
    query_feature: torch.Tensor,
    gallery_features: torch.Tensor,
    gallery_labels: torch.Tensor,
    gallery_paths: list[str],
    class_names: list[str],
    top_k: int = 10,
) -> list[dict]:
    """Find top-K most similar images to query."""
    # Cosine similarity
    similarities = torch.mm(query_feature, gallery_features.t()).squeeze(0)

    # Top-K
    topk_sim, topk_idx = similarities.topk(top_k)

    results = []
    for sim, idx in zip(topk_sim.tolist(), topk_idx.tolist()):
        label_idx = gallery_labels[idx].item()
        results.append({
            "path": gallery_paths[idx],
            "similarity": sim,
            "label": class_names[label_idx],
            "label_idx": label_idx,
        })

    return results


def visualize_results(
    query_path: str,
    results: list[dict],
    save_path: str | None = None,
):
    """Visualize query and retrieval results using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
    except ImportError:
        print("matplotlib not available. Showing text results only.")
        print(f"\nQuery: {query_path}")
        for i, r in enumerate(results):
            print(f"  #{i+1}: {r['label']} (sim={r['similarity']:.4f}) - {r['path']}")
        return

    n_results = len(results)
    fig, axes = plt.subplots(1, n_results + 1, figsize=(3 * (n_results + 1), 3.5))

    # Query image
    query_img = Image.open(query_path).convert("RGB")
    axes[0].imshow(query_img)
    axes[0].set_title("QUERY", fontsize=12, fontweight="bold", color="red")
    axes[0].axis("off")

    # Results
    for i, r in enumerate(results):
        img = Image.open(r["path"]).convert("RGB")
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(
            f"#{i+1}: {r['label']}\n{r['similarity']:.3f}",
            fontsize=9,
        )
        axes[i + 1].axis("off")

    plt.suptitle("SatLetter Retrieval Results", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Results saved to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="SatLetter Retrieval Demo")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--query", type=str, required=True,
                        help="Path to query image")
    parser.add_argument("--gallery", type=str, default="dataset/satellite_letters",
                        help="Path to gallery dataset")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of results to return")
    parser.add_argument("--save", type=str, default=None,
                        help="Save visualization to file")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto/cuda/cpu)")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Input image size")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    # Load model
    model = load_model(args.checkpoint, device)

    # Build gallery index
    print(f"\nBuilding gallery index from: {args.gallery}")
    gallery_features, gallery_labels, gallery_paths, class_names = build_gallery_index(
        model, args.gallery, device, args.image_size
    )
    print(f"Gallery size: {len(gallery_paths)} images, {len(class_names)} classes")

    # Extract query feature
    print(f"\nQuery image: {args.query}")
    query_feature = extract_query_feature(model, args.query, device, args.image_size)

    # Retrieve
    results = retrieve_similar(
        query_feature, gallery_features, gallery_labels, gallery_paths, class_names,
        top_k=args.top_k,
    )

    # Display results
    print(f"\nTop-{args.top_k} Results:")
    for i, r in enumerate(results):
        print(f"  #{i+1}: Letter '{r['label']}' | Similarity: {r['similarity']:.4f} | {os.path.basename(r['path'])}")

    # Visualize
    visualize_results(args.query, results, save_path=args.save)


if __name__ == "__main__":
    main()
