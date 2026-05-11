# Code Standards

## Language & Runtime

| Item | Standard |
|------|----------|
| Language | Python 3.10+ (type hints with `X | Y` syntax) |
| Runtime | Google Colab (Ubuntu + T4 GPU) |
| Package Manager | pip |
| Virtual Environment | Colab default (không cần local venv) |

## Project Layout

```
vit-ir/
├── src/                    # Core library (importable modules)
│   ├── models/             # Neural network architectures
│   ├── losses/             # Loss functions
│   ├── data/               # Dataset loaders & transforms
│   └── utils/              # Evaluation metrics & helpers
├── scripts/                # Standalone scripts (pipeline, training, eval)
├── docs/                   # Documentation
├── notebook/               # Jupyter notebooks
├── train.py                # Training entry point
├── demo.py                 # Inference demo
└── requirements.txt        # Dependencies
```

### Conventions

- **Core library** (`src/`): Importable modules, không chứa CLI logic
- **Entry points** (root): `train.py`, `demo.py` — chứa `argparse` + `main()`
- **Scripts** (`scripts/`): Standalone scripts, prefix số thứ tự (`01_`, `02_`, ...)
- **Notebooks** (`notebook/`): Colab notebooks, versioned (`_v1`, `_v2`)

## Coding Standards

### Type Hints

Sử dụng Python 3.10+ union syntax:

```python
# ✅ Correct
def forward(self, x: torch.Tensor) -> torch.Tensor: ...
def get_data(self) -> tuple[torch.Tensor, torch.Tensor] | None: ...
embed_dim: int | None = None

# ❌ Avoid
from typing import Optional, Tuple, Union
def get_data(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]: ...
```

### Docstrings

Google-style docstrings cho tất cả public classes và functions:

```python
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
```

### Module Structure

Mỗi module file bắt đầu bằng docstring mô tả:

```python
"""
Loss functions for IRT training
================================
Implements:
  - Contrastive Loss with margin (Eq. 1 from paper)
  - KoLeo Loss: Differential entropy regularization (Eq. 5)
  - Combined IRT Loss: L = L_contrastive + λ * L_KoLeo
  - Cross-Batch Memory (XBM) queue
"""
```

### Import Order

1. Standard library (`os`, `sys`, `argparse`, ...)
2. Third-party (`torch`, `numpy`, `PIL`, ...)
3. Local imports (`src.models`, `src.losses`, ...)

```python
import os
import sys

import torch
import torch.nn as nn
import numpy as np

from src.models.backbone import build_model
from src.losses.irt_losses import IRTLoss
```

## PyTorch Conventions

### Model Design

```python
class IRTModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Register submodules as attributes
        self.backbone = timm.create_model(...)
        self.projector = nn.Linear(...)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main forward pass — always returns (B, D) descriptors
        ...

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # Inference-only method
        self.eval()
        return self.forward(x)
```

### Loss Design

Loss functions trả về dictionary:

```python
def forward(self, embeddings, labels, ...) -> dict[str, torch.Tensor]:
    return {
        "total": total_loss,       # For backward()
        "contrastive": ...,         # For logging (detached)
        "koleo": ...,               # For logging (detached)
    }
```

### DataLoader Defaults

System-aware defaults cho Colab compatibility:

```python
# Auto-detect workers: min(cpu_count - 1, cpu_count), ít nhất 0
# Auto-detect pin_memory: True chỉ khi CUDA available
loader = get_dataloader(dataset, batch_size=64, num_workers=None, pin_memory=None)
```

### Checkpoint Format

```python
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "metrics": metrics,          # R@1, R@5, R@10, mAP
    "args": vars(args),          # Training config
}, path)
```

## Evaluation Standards

### Metric Naming

| Metric | Key | Range |
|--------|-----|-------|
| Recall@1 | `R@1` | 0-100% |
| Recall@5 | `R@5` | 0-100% |
| Recall@10 | `R@10` | 0-100% |
| Mean Average Precision | `mAP` | 0-100% |

### Feature Extraction

- Luôn L2-normalize embeddings trước khi compute similarity
- Cosine similarity = dot product (vì đã normalize)
- Dùng `@torch.no_grad()` cho evaluation

## Error Handling

### NaN Guard (Training)

```python
if torch.isnan(losses["total"]):
    print(f"  ⚠️  NaN loss at batch {batch_idx+1}, skipping...")
    optimizer.zero_grad()
    continue
```

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Missing Dataset

```python
if not os.path.isdir(split_dir):
    raise FileNotFoundError(f"Dataset directory not found: {split_dir}")
```

## CLI Interface

### Argument Naming

- `--dataset`: lowercase, underscore-separated (`emnist`, `cross_domain`)
- `--pretrained_from`: đường dẫn file
- `--eval_every`, `--save_every`: số epoch

### Device Auto-detection

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

## Version Control

### Gitignored

- `dataset/` — Generated data (large files)
- `checkpoints/` — Trained models (large files)
- `__pycache__/` — Python bytecode
- `*.pyc` — Compiled Python

### Tracked

- Tất cả source code (`src/`, `scripts/`, `train.py`, `demo.py`)
- Documentation (`docs/`, `README.md`)
- Notebooks (`notebook/`)
- Configuration (`requirements.txt`, `.gitignore`)
