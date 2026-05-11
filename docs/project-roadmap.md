# Project Roadmap

## Timeline Overview

```
Week 1          Week 2          Week 3          Week 4
────────────    ────────────    ────────────    ────────────
Dataset         Baseline        Fine-tune       Ablation
Collection      + EMNIST        + Evaluation    + Report
```

## Milestones

### ✅ Phase 1 — Dataset Pipeline (Week 1)

| Task | Status | Notes |
|------|--------|-------|
| EMNIST Letters download (145K images) | ✅ Complete | `scripts/01_download_emnist.py` |
| Rendered fonts generation (15.6K images) | ✅ Complete | `scripts/02_generate_rendered_fonts.py` |
| NASA satellite scraping (71 images) | ✅ Complete | `scripts/03_scrape_nasa_landsat.py` |
| Google Earth download helpers | ✅ Complete | `scripts/03c_download_google_earth.py` |
| Satellite augmentation pipeline | ✅ Complete | `scripts/04_augment_satellite.py` |
| Dataset verification | ✅ Complete | `scripts/05_verify_dataset.py` |
| Master pipeline runner | ✅ Complete | `scripts/run_pipeline.py` |
| Colab setup & integration | ✅ Complete | `scripts/colab_setup.py` |

### ✅ Phase 2 — Core Implementation (Week 2)

| Task | Status | Notes |
|------|--------|-------|
| IRTModel (DeiT-Small backbone) | ✅ Complete | `src/models/backbone.py` |
| GeM pooling | ✅ Complete | Learnable exponent |
| Contrastive loss | ✅ Complete | Margin-based, XBM support |
| KoLeo loss | ✅ Complete | NaN-safe implementation |
| Cross-Batch Memory (XBM) | ✅ Complete | FIFO queue |
| Training loop | ✅ Complete | NaN guard + gradient clipping |
| Evaluation metrics (R@K, mAP) | ✅ Complete | `src/utils/evaluation.py` |
| Stage 1 training (EMNIST) | ✅ Complete | `scripts/train_stage1.sh` |

### ✅ Phase 3 — Cross-Domain (Week 3)

| Task | Status | Notes |
|------|--------|-------|
| CrossDomainDataset | ✅ Complete | Unified labels + oversampling |
| Backbone freezing (6/12 layers) | ✅ Complete | `model.freeze_layers()` |
| Stage 2 training (cross-domain) | ✅ Complete | `scripts/train_stage2.sh` |
| Cross-domain evaluation | ✅ Complete | `scripts/evaluate_cross_domain.py` |
| Demo CLI | ✅ Complete | `demo.py` |

### 🔄 Phase 4 — Analysis & Report (Week 4)

| Task | Status | Notes |
|------|--------|-------|
| Ablation studies | 🔄 In Progress | Pretrain source, loss function, pooling |
| t-SNE/UMAP visualization | 📋 Planned | Embedding space visualization |
| Results analysis | 🔄 In Progress | Per-class breakdown, confusion matrix |
| Final report | 📋 Planned | Báo cáo đồ án |
| Demo notebook (Colab) | ✅ Complete | `notebook/vit_ir_ptit_v2.ipynb` |

## Future Improvements

### Short-term (nếu có thêm thời gian)

- [ ] Multi-scale feature extraction (different crop sizes)
- [ ] Test-time augmentation (TTA) cho retrieval
- [ ] PCA whitening post-processing
- [ ] Web-based demo (Gradio/Streamlit)

### Long-term (hướng nghiên cứu mở rộng)

- [ ] Scale lên backbone lớn hơn (DeiT-Base, 87M params)
- [ ] Dataset lớn hơn (Google Earth Engine API → hàng ngàn ảnh)
- [ ] Region-level retrieval (không chỉ image-level)
- [ ] Self-supervised pretraining (DINO, MAE)
- [ ] Real-time indexing với FAISS hoặc ScaNN

## Ablation Study Matrix

| Experiment | Variables | Status |
|-----------|----------|--------|
| Pretrain source | ImageNet only vs EMNIST vs EMNIST+Fonts | 📋 Planned |
| Loss function | Contrastive only vs Contrastive+KoLeo | 📋 Planned |
| Pooling method | CLS vs Avg vs GeM | 📋 Planned |
| Descriptor dim | 384 (native) vs 128 (PCA) | 📋 Planned |
| Backbone freeze | 0 vs 3 vs 6 vs 9 layers frozen | 📋 Planned |
| Satellite oversample | 5x vs 10x vs 20x | 📋 Planned |
| XBM size | 0 vs 1024 vs 2048 vs 4096 | 📋 Planned |
