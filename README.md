# OmniModel: Lightweight Dual-Encoder for Vision-Language Tasks 🧠👁️

**OmniModel** là một kiến trúc Dual-Encoder (Vision & Language) được xây dựng và huấn luyện từ đầu (from scratch) trên tập dữ liệu COCO. Mô hình sử dụng cơ chế Contrastive Learning (tương tự CLIP) kết hợp với Knowledge Distillation để đạt hiệu suất cao với kích thước nhỏ gọn.

![Architecture](https://img.shields.io/badge/Architecture-Dual%20Encoder-blue)
![Backbone](https://img.shields.io/badge/Vision-ConvNeXt%20Tiny-green)
![Backbone](https://img.shields.io/badge/Text-Custom%20Transformer-orange)

## 🌟 Tính năng chính
- **Vision Encoder:** ConvNeXt-Tiny tùy chỉnh, trích xuất đặc trưng hình ảnh mạnh mẽ.
- **Text Encoder:** Transformer Encoder (6 layers, 8 heads), hiểu ngữ nghĩa văn bản tiếng Anh.
- **Joint Embedding:** Không gian vector chung (512 dimensions) cho cả ảnh và chữ.
- **Portable:** Dễ dàng tách rời để làm Backbone cho Segmentation, Detection hoặc Image Retrieval.

## 📂 Cấu trúc dự án
Repo này chứa source code huấn luyện và SDK suy luận:

```text
OmniModel/
├── omni_model_release/      # SDK đóng gói để sử dụng ngay
│   ├── weights/             # (Cần tải weights bỏ vào đây)
│   ├── interface.py         # Cổng giao tiếp chính
│   └── config/              # Config model & tokenizer
├── src/                     # Source code gốc (Training core)
├── configs/                 # Cấu hình huấn luyện
└── train_clip.py            # Script huấn luyện chính
