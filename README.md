# OmniModel: Lightweight Vision-Language Dual Encoder 🧠

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights-orange)](https://huggingface.co/Bigkatoan/OmniModel)

**OmniModel** is a lightweight, open-source Dual-Encoder architecture designed for Vision-Language tasks. Trained from scratch on the COCO dataset using Contrastive Learning and Knowledge Distillation, it provides a shared embedding space (512-dim) for both images and text.

This library is optimized to serve as a **pre-trained backbone** for downstream tasks such as Semantic Segmentation, Image Retrieval, and Zero-shot Classification, specifically targeting edge devices where model size and inference speed are critical.

---

## 📚 Table of Contents
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Model Architecture](#-model-architecture)
- [Advanced Usage](#-advanced-usage)
  - [Image Retrieval](#1-image-retrieval)
  - [Zero-shot Classification](#2-zero-shot-classification)
  - [Backbone Extraction (Segmentation/Detection)](#3-backbone-extraction-for-downstream-tasks)
- [Weights Management](#-weights-management)
- [Training](#-training-reproduction)
- [License](#-license)

---

## 🌟 Key Features

* **Dual-Tower Architecture:** Separate encoders for Vision (ConvNeXt-Tiny) and Language (Transformer), projected into a unified embedding space.
* **Lightweight & Fast:** Designed for efficiency, making it suitable for real-time applications.
* **Hybrid Weights Loading:** Automatically fetches pre-trained weights from Hugging Face Hub if local weights are missing.
* **Downstream Ready:** Easily extracts the Vision Backbone (with multi-scale features) for UNet, FPN, or Mask-RCNN integration.

---

## 📦 Installation

### Option 1: Install via pip (Recommended)
You can install the package directly from the source or via PyPI (once published).

```bash
# Install from local directory
pip install .

# OR install directly from GitHub
pip install git+[https://github.com/Bigkatoan/OmniModel.git](https://github.com/Bigkatoan/OmniModel.git)
```

### Option 2: Manual Installation (For Developers)
```bash
git clone [https://github.com/Bigkatoan/OmniModel.git](https://github.com/Bigkatoan/OmniModel.git)
cd OmniModel
pip install -r requirements.txt
```

---

## 🚀 Quick Start

Here is a minimal example to verify the installation and model inference.

```python
from omni_model.interface import OmniModel

# Initialize the model (automatically downloads weights on first run)
model = OmniModel(device='cuda') # Use 'cpu' if CUDA is unavailable

# 1. Encode Text
text_emb = model.encode_text("a photo of a black dog")

# 2. Encode Image (Accepts file path or numpy array)
img_emb = model.encode_image("data/dog.jpg")

# 3. Compute Similarity (Dot Product)
# Since embeddings are normalized, dot product equals cosine similarity.
score = (img_emb @ text_emb.T).item()

print(f"Similarity Score: {score:.4f}")
```

---

## 🏗 Model Architecture

OmniModel utilizes an asymmetric dual-encoder design:

| Component | Architecture | Specifications | Output Shape |
| :--- | :--- | :--- | :--- |
| **Vision Encoder** | **ConvNeXt-Tiny** | Customized depth `[3, 3, 9, 3]`. Optimized for feature extraction. | `[B, 768, 7, 7]` (Last Map) |
| **Text Encoder** | **Transformer** | 6 Layers, 8 Heads, 512 Hidden Dim. | `[B, 512]` (CLS Token) |
| **Projection** | **Linear** | Maps vision features (768) to shared space (512). | `[B, 512]` |

---

## 📖 Advanced Usage

### 1. Image Retrieval
Find the most relevant image from a gallery based on a text query.

```python
import torch

# Define gallery and query
image_paths = ["assets/cat.jpg", "assets/dog.jpg", "assets/car.jpg"]
query = "a cute puppy"

# Encode query
query_vec = model.encode_text(query)

# Encode gallery (Batch processing recommended for large scale)
gallery_vecs = torch.cat([model.encode_image(p) for p in image_paths])

# Compute scores
scores = (query_vec @ gallery_vecs.T).squeeze()
best_idx = torch.argmax(scores).item()

print(f"Best match: {image_paths[best_idx]} (Score: {scores[best_idx]:.4f})")
```

### 2. Zero-shot Classification
Classify images without training by comparing image embeddings with class description embeddings.

```python
labels = ["person", "bicycle", "car", "motorcycle", "airplane"]
img_vec = model.encode_image("assets/test_image.jpg")

# Encode all labels
label_vecs = torch.cat([model.encode_text(f"a photo of a {l}") for l in labels])

# Calculate Softmax probabilities
logits = (img_vec @ label_vecs.T).squeeze()
probs = logits.softmax(dim=0)

for label, prob in zip(labels, probs):
    print(f"{label}: {prob:.1%}")
```

### 3. Backbone Extraction (for Downstream Tasks)
For tasks like **Semantic Segmentation** (e.g., Prompt-based UNet), you can extract the pre-trained vision backbone.

```python
import torch

# 1. Get the backbone
backbone = model.get_backbone()

# 2. Create a dummy input
dummy_input = torch.randn(1, 3, 224, 224).to(model.device)

# 3. Forward pass
# The backbone returns a list of feature maps from 4 stages (useful for FPN)
features = backbone(dummy_input)

for i, feat in enumerate(features):
    print(f"Stage {i} output shape: {feat.shape}")
    # Stage 0: [1, 96, 56, 56]
    # Stage 1: [1, 192, 28, 28]
    # Stage 2: [1, 384, 14, 14]
    # Stage 3: [1, 768, 7, 7]
```

---

## ☁️ Weights Management

OmniModel supports **Hybrid Mode** for weight loading:

1.  **Local Priority:** The system checks `omni_model/weights/` first. If `vision_encoder.pth`, `text_encoder.pth`, and `vision_proj.pth` exist, they are loaded immediately (Offline Mode).
2.  **Auto-Download:** If local files are missing, the system automatically downloads the latest weights from the [Hugging Face Hub](https://huggingface.co/Bigkatoan/OmniModel) and caches them.

**Manual Setup (Offline):**
If you are working in an air-gapped environment, download weights manually and place them as follows:
```text
omni_model/
└── weights/
    ├── vision_encoder.pth
    ├── text_encoder.pth
    └── vision_proj.pth
```

---

## 🛠 Training (Reproduction)

To reproduce the training results (Contrastive Learning + Distillation):

1.  **Prepare Data:** Download [COCO 2017](https://cocodataset.org/) dataset.
2.  **Configuration:** Update paths in `configs/text_pretrain.yaml`.
3.  **Run Training:**
    ```bash
    python train_clip.py
    ```

The training script supports **Gradient Accumulation** (for large effective batch sizes on limited VRAM) and **Knowledge Distillation** (using OpenAI CLIP as the teacher).

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

**Developed by Bigkatoan.**
