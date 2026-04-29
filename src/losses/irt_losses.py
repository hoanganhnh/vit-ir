"""
Loss functions for IRT training
================================
Implements:
  - Contrastive Loss with margin (Eq. 1 from paper)
  - KoLeo Loss: Differential entropy regularization (Eq. 5)
  - Combined IRT Loss: L = L_contrastive + λ * L_KoLeo
  - Cross-Batch Memory (XBM) queue
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Contrastive loss with margin for metric learning.

    Maximizes similarity between positive pairs and minimizes similarity
    between negative pairs that exceed the margin threshold.

    L = (1/N) Σ_i [ Σ_{j:y_i=y_j} [1 - z_i·z_j]_+
                   + Σ_{j:y_i≠y_j} [z_i·z_j - β]_+ ]

    Args:
        margin: Negative pair margin β (default: 0.5)
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        memory_embeddings: torch.Tensor | None = None,
        memory_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute contrastive loss.

        Args:
            embeddings: L2-normalized embeddings (B, D)
            labels: Class labels (B,)
            memory_embeddings: Optional cross-batch memory embeddings (M, D)
            memory_labels: Optional cross-batch memory labels (M,)

        Returns:
            Scalar loss value
        """
        # Combine with memory if available
        if memory_embeddings is not None and memory_labels is not None:
            all_embeddings = torch.cat([embeddings, memory_embeddings], dim=0)
            all_labels = torch.cat([labels, memory_labels], dim=0)
        else:
            all_embeddings = embeddings
            all_labels = labels

        B = embeddings.size(0)

        # Compute similarity matrix: (B, B+M)
        sim_matrix = torch.mm(embeddings, all_embeddings.t())  # cosine similarity

        # Create masks for positive and negative pairs
        label_eq = labels.unsqueeze(1) == all_labels.unsqueeze(0)  # (B, B+M)

        # Exclude self-similarity (diagonal for batch part)
        eye_mask = torch.zeros(B, all_embeddings.size(0), dtype=torch.bool, device=embeddings.device)
        eye_mask[:, :B] = torch.eye(B, dtype=torch.bool, device=embeddings.device)
        label_eq = label_eq & ~eye_mask

        pos_mask = label_eq
        neg_mask = ~label_eq & ~eye_mask

        # Positive loss: minimize distance between positives
        pos_sim = sim_matrix[pos_mask]
        if pos_sim.numel() > 0:
            pos_loss = F.relu(1.0 - pos_sim).mean()
        else:
            pos_loss = torch.tensor(0.0, device=embeddings.device)

        # Negative loss: push negatives beyond margin
        neg_sim = sim_matrix[neg_mask]
        if neg_sim.numel() > 0:
            neg_loss = F.relu(neg_sim - self.margin).mean()
        else:
            neg_loss = torch.tensor(0.0, device=embeddings.device)

        return pos_loss + neg_loss


class KoLeoLoss(nn.Module):
    """KoLeo (Kozachenko-Leonenko) differential entropy loss.

    Maximizes the distance between each embedding and its nearest neighbor,
    encouraging uniform distribution on the hypersphere. Prevents feature collapse.

    L_KoLeo = -(1/N) Σ_i log(ρ_i)
    where ρ_i = min_{j≠i} ||z_i - z_j||

    From: Sablayrolles et al. "Spreading vectors for similarity search" (ICLR 2019)
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute KoLeo loss.

        Args:
            embeddings: L2-normalized embeddings (B, D)

        Returns:
            Scalar loss value
        """
        # Compute pairwise distances
        # ||z_i - z_j||^2 = ||z_i||^2 + ||z_j||^2 - 2*z_i·z_j
        # Since embeddings are L2-normalized: = 2 - 2*z_i·z_j
        dot_products = torch.mm(embeddings, embeddings.t())  # (B, B)

        # Set diagonal to -inf so self is never nearest neighbor
        B = embeddings.size(0)
        dot_products.fill_diagonal_(-float("inf"))

        # Nearest neighbor distance: min_j ||z_i - z_j|| = sqrt(2 - 2*max_j(z_i·z_j))
        max_sim, _ = dot_products.max(dim=1)  # (B,)
        nn_dist = torch.sqrt(2.0 - 2.0 * max_sim + self.eps)  # (B,)

        # KoLeo loss: negative mean log of nearest neighbor distances
        loss = -torch.log(nn_dist + self.eps).mean()

        return loss


class IRTLoss(nn.Module):
    """Combined IRT loss: Contrastive + λ * KoLeo.

    L = L_contrastive + λ * L_KoLeo

    Args:
        margin: Contrastive loss margin β (default: 0.5)
        lambda_koleo: KoLeo regularization strength λ (default: 0.7)
        use_koleo: Whether to include KoLeo regularization (False = IRTL, True = IRTR)
    """

    def __init__(
        self,
        margin: float = 0.5,
        lambda_koleo: float = 0.7,
        use_koleo: bool = True,
    ):
        super().__init__()
        self.contrastive = ContrastiveLoss(margin=margin)
        self.koleo = KoLeoLoss()
        self.lambda_koleo = lambda_koleo
        self.use_koleo = use_koleo

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        memory_embeddings: torch.Tensor | None = None,
        memory_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Returns:
            Dictionary with 'total', 'contrastive', and 'koleo' loss values
        """
        loss_contrastive = self.contrastive(
            embeddings, labels, memory_embeddings, memory_labels
        )

        if self.use_koleo:
            loss_koleo = self.koleo(embeddings)
            total = loss_contrastive + self.lambda_koleo * loss_koleo
        else:
            loss_koleo = torch.tensor(0.0, device=embeddings.device)
            total = loss_contrastive

        return {
            "total": total,
            "contrastive": loss_contrastive.detach(),
            "koleo": loss_koleo.detach(),
        }


class CrossBatchMemory(nn.Module):
    """Cross-Batch Memory (XBM) queue for metric learning.

    Maintains a FIFO queue of embeddings from previous batches to provide
    more negative pairs for contrastive learning.

    From: Wang et al. "Cross-Batch Memory for Embedding Learning" (CVPR 2020)

    Args:
        memory_size: Maximum number of embeddings to store
        embed_dim: Dimensionality of embeddings
    """

    def __init__(self, memory_size: int, embed_dim: int):
        super().__init__()
        self.memory_size = memory_size

        # Register as buffers (saved with model, moved to device, but not optimized)
        self.register_buffer("embeddings", torch.zeros(memory_size, embed_dim))
        self.register_buffer("labels", torch.full((memory_size,), -1, dtype=torch.long))
        self.register_buffer("ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("is_full", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def enqueue(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """Add new embeddings to the memory queue."""
        batch_size = embeddings.size(0)
        ptr = self.ptr.item()

        if ptr + batch_size <= self.memory_size:
            self.embeddings[ptr : ptr + batch_size] = embeddings.detach()
            self.labels[ptr : ptr + batch_size] = labels.detach()
        else:
            # Wrap around
            overflow = (ptr + batch_size) - self.memory_size
            self.embeddings[ptr:] = embeddings[: self.memory_size - ptr].detach()
            self.labels[ptr:] = labels[: self.memory_size - ptr].detach()
            self.embeddings[:overflow] = embeddings[self.memory_size - ptr :].detach()
            self.labels[:overflow] = labels[self.memory_size - ptr :].detach()
            self.is_full.fill_(True)

        self.ptr.fill_((ptr + batch_size) % self.memory_size)

    def get(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Get current memory contents. Returns None if empty."""
        if not self.is_full and self.ptr.item() == 0:
            return None

        if self.is_full:
            return self.embeddings, self.labels
        else:
            size = self.ptr.item()
            return self.embeddings[:size], self.labels[:size]
