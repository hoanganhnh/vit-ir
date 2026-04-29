"""
Evaluation metrics for image retrieval
=======================================
Implements:
  - Recall@K
  - Mean Average Precision (mAP)
  - Feature extraction utilities
  - t-SNE / UMAP visualization
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


@torch.no_grad()
def extract_all_features(
    model,
    dataloader,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract features and labels for entire dataset.

    Returns:
        features: (N, D) L2-normalized feature matrix
        labels: (N,) label vector
    """
    model.eval()
    all_features = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device)
        features = model(images)  # (B, D), already L2-normalized
        all_features.append(features.cpu())
        all_labels.append(labels)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return features, labels


def compute_recall_at_k(
    query_features: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_features: torch.Tensor,
    gallery_labels: torch.Tensor,
    k_values: list[int] = [1, 5, 10],
    exclude_self: bool = True,
) -> dict[str, float]:
    """Compute Recall@K for image retrieval.

    Args:
        query_features: (Nq, D) query descriptors
        query_labels: (Nq,) query labels
        gallery_features: (Ng, D) gallery descriptors
        gallery_labels: (Ng,) gallery labels
        k_values: List of K values to compute
        exclude_self: Exclude exact matches (when query = gallery)

    Returns:
        Dictionary mapping 'R@K' to float values
    """
    # Cosine similarity matrix (features are L2-normalized)
    sim_matrix = torch.mm(query_features, gallery_features.t())  # (Nq, Ng)

    if exclude_self and query_features.data_ptr() == gallery_features.data_ptr():
        # Same set: exclude self by setting diagonal to -inf
        sim_matrix.fill_diagonal_(-float("inf"))

    results = {}
    max_k = max(k_values)

    # Get top-K indices
    _, topk_indices = sim_matrix.topk(max_k, dim=1)  # (Nq, max_k)
    topk_labels = gallery_labels[topk_indices]  # (Nq, max_k)

    # Check matches
    matches = topk_labels == query_labels.unsqueeze(1)  # (Nq, max_k)

    for k in k_values:
        # Recall@K: fraction of queries with at least one correct match in top-K
        recall = matches[:, :k].any(dim=1).float().mean().item()
        results[f"R@{k}"] = recall * 100.0

    return results


def compute_map(
    query_features: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_features: torch.Tensor,
    gallery_labels: torch.Tensor,
    exclude_self: bool = True,
) -> float:
    """Compute Mean Average Precision (mAP).

    Args:
        query_features: (Nq, D) query descriptors
        query_labels: (Nq,) query labels
        gallery_features: (Ng, D) gallery descriptors
        gallery_labels: (Ng,) gallery labels

    Returns:
        mAP score (0-100)
    """
    sim_matrix = torch.mm(query_features, gallery_features.t())

    if exclude_self and query_features.data_ptr() == gallery_features.data_ptr():
        sim_matrix.fill_diagonal_(-float("inf"))

    # Sort by similarity (descending)
    sorted_indices = sim_matrix.argsort(dim=1, descending=True)
    sorted_labels = gallery_labels[sorted_indices]

    # Compute AP for each query
    aps = []
    for i in range(query_features.size(0)):
        matches = (sorted_labels[i] == query_labels[i]).float()
        n_relevant = matches.sum().item()

        if n_relevant == 0:
            continue

        # Precision at each position
        cumsum = matches.cumsum(dim=0)
        precision_at_k = cumsum / torch.arange(1, len(matches) + 1, device=matches.device).float()

        # Average precision
        ap = (precision_at_k * matches).sum() / n_relevant
        aps.append(ap.item())

    return (sum(aps) / len(aps) * 100.0) if aps else 0.0


def evaluate_retrieval(
    model,
    test_loader,
    device: str = "cuda",
    k_values: list[int] = [1, 5, 10],
) -> dict[str, float]:
    """Full evaluation pipeline: extract features → compute metrics.

    Returns:
        Dictionary with R@K and mAP metrics
    """
    features, labels = extract_all_features(model, test_loader, device)

    # Compute Recall@K (query = gallery for standard evaluation)
    recall_results = compute_recall_at_k(
        features, labels, features, labels, k_values, exclude_self=True
    )

    # Compute mAP
    map_score = compute_map(features, labels, features, labels, exclude_self=True)
    recall_results["mAP"] = map_score

    return recall_results


def print_metrics(metrics: dict[str, float], prefix: str = ""):
    """Pretty-print evaluation metrics."""
    header = f"{prefix} " if prefix else ""
    parts = [f"{k}: {v:.2f}%" for k, v in metrics.items()]
    print(f"{header}| {' | '.join(parts)} |")
