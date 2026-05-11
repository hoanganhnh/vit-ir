# SatLetter: Satellite Letter-Shape Image Retrieval — Project Overview & PDR

## Overview

**SatLetter** là hệ thống truy xuất ảnh (image retrieval) sử dụng Vision Transformers để tìm kiếm ảnh vệ tinh có hình dạng giống chữ cái. Dự án áp dụng phương pháp IRT (Image Retrieval with Transformers) từ bài báo gốc của El-Nouby et al. (ICML 2022), mở rộng cho bài toán cross-domain retrieval giữa ảnh chữ cái viết tay/rendered fonts và ảnh vệ tinh thật.

## Product Development Requirements

### Problem Statement

Xây dựng hệ thống **image-to-image retrieval** cho ảnh vệ tinh:

- **Input**: Ảnh chữ cái (rendered font, ảnh viết tay EMNIST, hoặc ảnh vệ tinh mẫu)
- **Output**: Top-K ảnh vệ tinh có hình dạng tự nhiên tương tự nhất từ cơ sở dữ liệu

**Ví dụ**: Input ảnh chữ "S" → Trả về ảnh vệ tinh con sông uốn cong hình chữ S.

### Stakeholders

| Stakeholder | Vai trò |
|-------------|---------|
| Sinh viên PTIT | Phát triển hệ thống, viết báo cáo |
| Giảng viên | Đánh giá đồ án |
| Cộng đồng nghiên cứu | Tham khảo phương pháp cross-domain retrieval |

### Scope

| In Scope | Out of Scope |
|----------|-------------|
| Cross-domain image retrieval pipeline | Real-time production deployment |
| DeiT-Small backbone + contrastive + KoLeo loss | Large-scale distributed training |
| 3-tier dataset pipeline (EMNIST → Fonts → Satellite) | Web interface / API server |
| Evaluation metrics: Recall@K, mAP | Video retrieval |
| CLI demo (query → top-K results) | Fine-grained localization |

### Success Criteria

| Metric | Mục tiêu | Stretch |
|--------|----------|---------|
| Cross-domain R@1 (EMNIST → Satellite) | >40% | >55% |
| Cross-domain R@5 (EMNIST → Satellite) | >65% | >80% |
| Cross-domain R@1 (Fonts → Satellite) | >45% | >60% |
| Same-domain R@1 (Satellite → Satellite) | >60% | >75% |
| Demo: đúng chữ cái trong top-3 | Nhất quán | Top-1 đúng |

### Constraints

| Constraint | Details |
|-----------|---------|
| GPU | Google Colab T4 (15GB VRAM) |
| Timeline | 4 tuần |
| Storage | Colab ~100GB |
| Framework | PyTorch + timm |

### Deliverables

1. Codebase Python (PyTorch) chạy trên Google Colab
2. SatFonts synthetic dataset (~6,240 ảnh) + NASA evaluation set (71 raw + augmented)
3. Trained models + cross-domain evaluation results
4. Báo cáo đồ án
5. Demo notebook (input ảnh → retrieval results)

## References

- [Training Vision Transformers for Image Retrieval](paper/) — El-Nouby et al., ICML 2022
- [SatLetter Retrieval Design](plans/2026-04-29-satletter-retrieval-design.md)
- [Cross-Domain Retrieval Design v2](plans/2026-05-08-cross-domain-retrieval-design.md)
