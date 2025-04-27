# Ultralytics Fork for Multispectral & Change-Detection

This repository is a fork of the [Ultralytics](https://github.com/ultralytics/ultralytics) package, extended to support:

- **Multispectral & Remote-Sensing Inputs**  
  – Full compatibility with images having arbitrary channel counts (e.g. RGB+NIR, multi-band satellite)
    - supporting tiff files with n bands as a new file type
    – Missing input-channel weights are initialized via Xavier uniform when loading a pretrained checkpoint  


- **Siamese / Dual-Stream Architectures**  
  – Built-in support for Siamese YOLO models geared toward bi-temporal change detection  
  – Example config included for a Siamese YOLOv9e; additional dual-stream configs coming soon  

- **Tested Models & Tasks**  
  – **Classification & Detection** on multispectral data: YOLOv8, YOLOv9, and YOLOv10  
  – Other tasks (segmentation, pose) may work but are currently untested or brokene
  - **Siamese** object level change detection for YOLOv9e. 

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

![siamese_schema.png](siamese_schema.png)

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
## configuration

### mulitlayer 

### 3. Configuring the Model for 6-Channel Input

**A) Fine-tune from pretrained RGB weights**  
```python
from ultralytics_MB import YOLO

# 1. Load a pretrained YOLOv8n (.pt) checkpoint
model = YOLO('yolov8n.pt')

# 2. Tell it to expect 6 input channels instead of 3
#    and only adjust the first conv layer’s weights
model.train(
    data='data.yaml',
    channels=6,           # input now has 6 bands (RGB@t1 + RGB@t2)
    adjust_layers=[0],    # remaps layer 0 weights from 3→6 channels
    bands = [0,1,2,3,4,5], # defines which bands should be loaded 

    # … other args (imgsz, epochs, batch, etc.)
)
```
- **`channels=6`** re-builds the very first conv kernel to accept 6 bands.  
- **`adjust_layers=[0]`** loads the RGB weights into both halves of that new 6-channel kernel, so you retain all pretrained features for timestamp 1 & 2.
- ** bands = [0,1,2,3,4,5]** defines which bands should be loaded 

---

**B) Train from scratch on 6 channels**  
```python
from ultralytics_MB import YOLO

# 1. Load an untrained YOLOv8n architecture (.yaml)
model = YOLO('custom_colo.yaml')

# 2. Directly train with 6 inputs—no weight remapping needed
model.train(
    data='data.yaml',
    channels=6,  
    bands = [0,1,2,3,4,5]
)
```
- Using the `.yaml` spec builds your network “from scratch” with the requested number of channels baked in.

### 3. Configuring the Siamese YOLOv9e-s (ES) for 6-Channel Dual-Stream Input

```python
from ultralytics_MB import YOLO

# 1. Load the Siamese-enabled “ES” spec — NOT the standard yolov9e.yaml!
model = YOLO('yolov9es.yaml', task='detect')

# 2. Tell it you have 6 input bands (RGB@t1 + RGB@t2)
#    and enable the Siamese dual-stream head
model.train(
    data='data.yaml',
    channels=6,         # network expects 6-band inputs
    bands=[0,1,2,3,4,5], # indexes of the band that should be loaded, first 
    dual_stream=True,   # turn on Siamese twin-stream processing
    # … other hyperparams go below
)
```

- **`yolov9es.yaml`** is the special “ES” version with the extra dual-stream modules baked in.  
- **`channels=6`** defines your two RGB timestamps as six input planes.  
- **`bands`** = [0,1,2,3,4,5] defines which bands should be loaded, first half corresponds to image one and the second to image 2 
- **`dual_stream=True`** wires the Siamese twin backbones and fusion head for change detection.

- **`imgsz`**, **`epochs`**, **`batch`**, **`single_cls`** are your usual YOLO training flags.  
- **`channels`** and **`dual_stream`** are the only extras to spin up the Siamese network on multi-band imagery.  


### `FUSION_METHOD`

The `FeatureFusionBlock` picks its fusion strategy from the `FUSION_METHOD` environment variable. If you don’t set it, it defaults to **`diff`** (element-wise subtraction).

#### Available modes

- **`add`**  
  Element-wise addition:  
  ```python
  out = x1 + x2
  ```

- **`diff`**  
  Element-wise difference:  
  ```python
  out = x1 - x2
  ```

- **`multiply`**  
  Element-wise multiplication:  
  ```python
  out = x1 * x2
  ```

- **`weighted`**  
  Learnable weighted sum. You must pass two floats via `params=[w1, w2]` when you construct the block:  
  ```python
  block = FeatureFusionBlock(params=[0.7, 0.3], in_channels=128)
  # out = 0.7*x1 + 0.3*x2
  ```

- **`attention`**  
  Squeeze-and-Excitation channel attention on (x1 + x2).

- **`cross_attention`**  
  QKV-style cross-attention between x1 (queries) and x2 (keys, values).

- **`cbam`**  
  CBAM (Channel + Spatial) attention on (x1 + x2).

- **`cbam_cross_attention`**  
  CBAM followed by QKV cross-attention.

#### How to set

Before creating your fusion block, export the choice in your shell:

```bash
export FUSION_METHOD=attention
```



