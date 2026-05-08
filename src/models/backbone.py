"""
IRT Backbone: DeiT-based feature extractor for image retrieval
==============================================================
Implements IRTO, IRTL, IRTR variants from the paper:
  "Training Vision Transformers for Image Retrieval" (El-Nouby et al., ICML 2022)

Key components:
  - DeiT-Small backbone (22M params)
  - CLS token as image descriptor (default)
  - Optional: Average/Max/GeM pooling
  - L2 normalization
  - Optional PCA dimensionality reduction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GeM(nn.Module):
    """Generalized Mean Pooling (GeM).

    Pools feature maps using a generalized mean with learnable exponent p.
    When p=1: average pooling. When p→∞: max pooling.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = True):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p)) if learnable else p
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) token features → (B, D) pooled."""
        return (
            x.clamp(min=self.eps)
            .pow(self.p)
            .mean(dim=1)
            .pow(1.0 / self.p)
        )


class IRTModel(nn.Module):
    """Image Retrieval with Transformers (IRT) model.

    Uses a Vision Transformer (DeiT) backbone to extract compact image descriptors
    for similarity-based image retrieval.

    Args:
        backbone_name: timm model name (default: 'deit_small_patch16_224')
        pooling: Feature aggregation method ('cls', 'avg', 'max', 'gem')
        normalize: Whether to L2-normalize output descriptors
        embed_dim: If set, project features to this dimensionality
        pretrained: Whether to use ImageNet-pretrained weights
    """

    def __init__(
        self,
        backbone_name: str = "deit_small_patch16_224",
        pooling: str = "cls",
        normalize: bool = True,
        embed_dim: int | None = None,
        pretrained: bool = True,
    ):
        super().__init__()

        # Load backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )

        # Get feature dimensionality
        self.feature_dim = self.backbone.embed_dim
        self.pooling_type = pooling
        self.normalize = normalize

        # Pooling layer
        if pooling == "gem":
            self.pool = GeM(p=3.0)
        elif pooling in ("cls", "avg", "max"):
            self.pool = None  # Handled in forward()
        else:
            raise ValueError(f"Unknown pooling: {pooling}. Use 'cls', 'avg', 'max', or 'gem'.")

        # Optional projection head
        self.projector = None
        self.output_dim = self.feature_dim
        if embed_dim is not None and embed_dim != self.feature_dim:
            self.projector = nn.Linear(self.feature_dim, embed_dim)
            self.output_dim = embed_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract raw token features from backbone.

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            Token features (B, N+1, D) where N = num patches, +1 for CLS
        """
        # timm DeiT forward_features returns (B, N+1, D) or (B, D) depending on version
        # We need the full token sequence for pooling options
        features = self.backbone.forward_features(x)

        # Handle different timm output formats
        if features.dim() == 2:
            # Already pooled (some timm versions) - unsqueeze for consistency
            return features.unsqueeze(1)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract image descriptors.

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            Descriptors (B, output_dim), L2-normalized if self.normalize=True
        """
        features = self.forward_features(x)  # (B, N+1, D)

        # Pooling
        if self.pooling_type == "cls":
            # Use CLS token (first token)
            descriptors = features[:, 0]  # (B, D)
        elif self.pooling_type == "avg":
            # Average pool over patch tokens (exclude CLS)
            descriptors = features[:, 1:].mean(dim=1)  # (B, D)
        elif self.pooling_type == "max":
            # Max pool over patch tokens
            descriptors = features[:, 1:].max(dim=1)[0]  # (B, D)
        elif self.pooling_type == "gem":
            # GeM pool over patch tokens
            descriptors = self.pool(features[:, 1:])  # (B, D)
        else:
            descriptors = features[:, 0]

        # Optional projection
        if self.projector is not None:
            descriptors = self.projector(descriptors)

        # L2 normalization
        if self.normalize:
            descriptors = F.normalize(descriptors, p=2, dim=-1)

        return descriptors

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features in evaluation mode (no grad)."""
        self.eval()
        return self.forward(x)

    def freeze_layers(self, num_layers: int = 6):
        """Freeze patch embedding and first N transformer blocks.

        DeiT-Small has 12 blocks. Freezing 6 means:
        - Layers 0-5: frozen (low-level features preserved from pretraining)
        - Layers 6-11: trainable (high-level adaptation for new domain)
        - CLS token, pos_embed, norm: always trainable

        Args:
            num_layers: Number of transformer blocks to freeze (0-12)
        """
        # Freeze patch embedding
        for p in self.backbone.patch_embed.parameters():
            p.requires_grad = False

        # Freeze first N transformer blocks
        for i in range(min(num_layers, len(self.backbone.blocks))):
            for p in self.backbone.blocks[i].parameters():
                p.requires_grad = False

        # Count trainable params
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"Frozen {num_layers} layers: {frozen:,} frozen, {trainable:,} trainable ({trainable/total*100:.1f}%)")



def build_model(
    backbone: str = "deit_small_patch16_224",
    pooling: str = "cls",
    embed_dim: int | None = None,
    pretrained: bool = True,
) -> IRTModel:
    """Factory function to build an IRT model.

    Available backbones (recommended):
        - 'deit_small_patch16_224'     (22M, 384-D, default)
        - 'deit_small_distilled_patch16_224'  (22M, 384-D, distilled)
        - 'deit_base_patch16_224'      (87M, 768-D)
        - 'deit_base_distilled_patch16_224'   (87M, 768-D, distilled)
        - 'vit_small_patch16_224'      (22M, 384-D)
    """
    model = IRTModel(
        backbone_name=backbone,
        pooling=pooling,
        normalize=True,
        embed_dim=embed_dim,
        pretrained=pretrained,
    )

    print(f"Built IRT model:")
    print(f"  Backbone:    {backbone}")
    print(f"  Pooling:     {pooling}")
    print(f"  Feature dim: {model.feature_dim}")
    print(f"  Output dim:  {model.output_dim}")
    print(f"  Params:      {sum(p.numel() for p in model.parameters()):,}")

    return model
