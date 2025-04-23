# Ultralytics Fork for Multispectral & Change-Detection

This repository is a fork of the [Ultralytics](https://github.com/ultralytics/ultralytics) package, extended to support:

- **Multispectral & Remote-Sensing Inputs**  
  – Full compatibility with images having arbitrary channel counts (e.g. RGB+NIR, multi-band satellite)  
  – Missing input-channel weights are initialized via Xavier uniform when loading a pretrained checkpoint  

- **Siamese / Dual-Stream Architectures**  
  – Built-in support for Siamese YOLO models geared toward bi-temporal change detection  
  – Example config included for a Siamese YOLOv9e; additional dual-stream configs coming soon  

- **Tested Models & Tasks**  
  – **Classification & Detection** on multispectral data: YOLOv8, YOLOv9, and YOLOv10  
  – Other tasks (segmentation, pose) may work but are currently untested or broken  

- **Usage Modes**  
  – Train from scratch on your own multispectral dataset  
  – Fine-tune from any Ultralytics pretrained checkpoint, automatically adapting input-layer dimensions  

> **Warning:** This fork is under active development and subject to breaking changes.  
> To avoid unexpected issues, pin your project to a specific release tag or commit hash.

---

## What Is YOLO?

**You Only Look Once (YOLO)** is a family of single-stage object detectors that perform localization and classification in one forward pass through a convolutional network.  
- The input image is divided into a grid; each cell predicts a set of bounding boxes and class probabilities simultaneously.  
- Modern YOLO versions (v8, v9, v10) use advanced backbones, feature-pyramid necks, and optimized heads to balance speed (dozens of FPS) with high accuracy.  
- Ultralytics’ implementation provides an easy CLI/API, out-of-the-box pretrained weights, and a consistent hyperparameter configuration.

---

## What Are Siamese Networks?

A **Siamese network** consists of two identical subnetworks with shared weights, processing two inputs in parallel. The key idea is to learn feature embeddings that highlight similarities or differences between the inputs, making Siamese architectures well suited for tasks such as:

- **Verification & Matching**: e.g., face or signature verification  
- **Change Detection**: comparing bi-temporal imagery to isolate new or altered features  

In this fork, we combine a Siamese backbone with YOLO’s detection head to focus on _change cues_ between a reference image (e.g., a base map) and a query image (e.g., a newly annotated or temporally updated scene).

---

## Simplified Comparison: YOLO vs. Siamese YOLO

> *[Insert your diagram or bullet-list here, showing two parallel backbones with shared weights vs. the single-backbone standard YOLO, and how feature-difference fusion feeds into the detection head.]*

| Feature                | Standard YOLO            | Siamese YOLO                       |
|------------------------|--------------------------|------------------------------------|
| Input                  | Single image             | Pair of images (reference + query) |
| Feature extractor      | One backbone             | Two synchronized backbones        |
| Fusion mechanism       | —                        | Difference or attention fusion     |
| Detection focus        | All objects              | Changes between inputs             |
| Use case               | Object detection         | Change detection & comparison      |

---

## Quick Start

1. **Install**  
   ```bash
   pip install git+https://github.com/your-org/ultralytics-multispectral.git
   ```
2. **parameters**
   please take a look at the two provied jupyter notebooks they introduce you to the setting of the hyper parameters
   
