
# 🏜️ Offroad Semantic Scene Segmentation — Team [0]

> **YOLO Hackathon 2026** · VIT Kondhwa, Pune · Organized by 23ventures · April 6, 2026
> **Track:** Segmentation Track — Duality AI · **Final mIoU: 0.4676**

**Team Members:** Saad Shaikh · Sairaj Shedge · Aviraj Virape

---

## 📌 Overview

This project tackles **semantic scene segmentation for Unmanned Ground Vehicles (UGVs)** operating in unstructured desert environments. We trained a **DINOv2 + ConvTranspose2d** segmentation model on the **Duality AI Falcon Synthetic Dataset** to classify every pixel into one of 10 terrain classes.

Real desert data is scarce and expensive to annotate — Duality AI's Falcon platform solves this by providing photorealistic synthetic images with pixel-perfect labels.

---

## 🧠 Model Architecture

| Component | Details |
|---|---|
| Backbone | DINOv2 (ViT-S/14) — frozen (22M params) |
| Decoder | ConvTranspose2d Head (384 → 256 → 128 → 64 → 10 channels) |
| Loss | 60% CrossEntropyLoss + 40% DiceLoss (class-weighted) |
| Optimizer | AdamW (lr=3e-4, weight_decay=0.01) |
| Scheduler | CosineAnnealingLR |
| Inference | Test-Time Augmentation (TTA) — 6 predictions (3 scales × original + hflip) |

> At epoch 20, the last 2 backbone transformer blocks were unfrozen for fine-tuning at lr=1e-5.

---

## 📊 Results

| Metric | Value |
|---|---|
| **Best Val mIoU (no TTA)** | 0.4628 |
| **Final mIoU (with TTA)** | **0.4676** |
| Training Loss (final) | 0.5592 |
| Validation Loss (final) | 0.63 |
| GPU | Tesla T4 |
| Training Time | ~3 hours (30 epochs, early stop) |

### Per-Class IoU

| Class | IoU | Status |
|---|---|---|
| Sky | 0.955 | ✅ Excellent |
| Trees | 0.672 | ✅ Good |
| Dry Grass | 0.606 | ✅ Good |
| Lush Bushes | 0.591 | ✅ Good |
| Landscape | 0.487 | 🟡 Average |
| Flowers | 0.444 | 🟡 Average |
| Dry Bushes | 0.326 | 🔴 Poor |
| Rocks | 0.268 | 🔴 Poor |
| Ground Clutter | 0.269 | 🔴 Poor |
| Logs | 0.059 | 🔴 Poor (severe class imbalance) |
| **Mean IoU** | **0.4676** | — |

---

## 🗂️ Dataset

| Property | Details |
|---|---|
| Source | Duality AI Falcon Platform |
| Environment | Desert (Multiple Locations) |
| Classes | 10 |
| Training Images | 2,857 |
| Validation Images | 317 |
| Image Resolution | 1920×1080 → resized to 504×504 |

### Segmentation Classes

| Class ID | Name | Description |
|---|---|---|
| 100 | Trees | Desert trees (e.g. Joshua trees) |
| 200 | Lush Bushes | Green/leafy bushes |
| 300 | Dry Grass | Dead/dry grass |
| 500 | Dry Bushes | Sparse dry bushes |
| 550 | Ground Clutter | Small rocks and debris |
| 600 | Flowers | Desert flowers |
| 700 | Logs | Fallen logs |
| 800 | Rocks | Rocks and boulders |
| 7100 | Landscape | General terrain/ground |
| 10000 | Sky | Sky region |

---

## 🚀 Setup & Usage

### 1. Clone & Open in Google Colab

```bash
git clone https://github.com/saad-aids/<repo-name>.git
```

Then open `hackathon_yolo_edition2_1_.py` in [Google Colab](https://colab.research.google.com/) with a **T4 GPU runtime**.

### 2. Install Dependencies

```bash
pip install torch torchvision albumentations segmentation-models-pytorch timm
```

Or run Cell 1–2 in the notebook which handles this automatically.

### 3. Dataset Setup

Place your dataset zip in Google Drive at:
```
MyDrive/duality_hackathon/dataset/Offroad_Segmentation_Training_Dataset.zip
```

The notebook will extract and verify the structure automatically.

### 4. Training

Run Cells 1–9 sequentially. The best model is auto-saved to:
```
MyDrive/duality_hackathon/weights/best_model.pth
```

Training uses early stopping with `patience=10`.

### 5. Inference (TTA)

```python
pred = tta_predict(model, img_rgb, device)
```

TTA applies 6 predictions (3 scales × original + horizontal flip) and averages probabilities before argmax.

---

## 📁 Project Structure

```
├── hackathon_yolo_edition2_1_.py   # Full pipeline (Colab notebook exported)
├── README.md                       # This file
└── (Google Drive)
    ├── weights/
    │   └── best_model.pth          # Trained model weights
    ├── outputs/                    # Test image predictions (colorized)
    └── runs/
        ├── training_curves.png     # Loss & mIoU plots
        ├── per_class_iou.png       # Per-class IoU bar chart
        └── failure_cases.png       # 4 worst validation predictions
```

---

## 🔧 Key Design Choices

- **Frozen backbone** → prevents overfitting on small synthetic dataset
- **Class-weighted loss** → combats severe class imbalance (Logs: very rare)
- **Progressive unfreezing** at epoch 20 → fine-tunes backbone at low lr
- **TTA (6-way ensemble)** → boosts mIoU by +0.0048 at zero training cost
- **CoarseDropout augmentation** → improves robustness to occluded regions

---

## 📎 References

- [DINOv2 — Meta AI](https://github.com/facebookresearch/dinov2)
- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- [albumentations](https://albumentations.ai/)
- [Duality AI Falcon Platform](https://www.duality.ai/)

---

*Built with ❤️ at YOLO Hackathon 2026, VIT Kondhwa, Pune*
