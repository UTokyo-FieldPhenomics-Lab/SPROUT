# SPROUT: A Scalable Diffusion Foundation Model for Multi-Crop Plant Phenotyping

Shuai Xiang, James Burridge, Shouyang Liu, Hao Lu, Tokihiro Fukatsu, Yinqiang Zheng, and Wei Guo

**[Paper](https://arxiv.org/abs/2603.27519) | [Pretrained Weights](#pretrained-weights) | [Pretraining](#pretraining-from-unlabeled-images) | [MCD-2.6M Dataset](https://huggingface.co/datasets/XIANG-Shuai/MCD-2.6m) | [Citation](#citation)**

SPROUT is a diffusion foundation model for multi-crop plant phenotyping. Pretrained on 2.6 million unlabeled open-field images, it supports plant organ segmentation, crop–weed parsing, monocular depth estimation, and yield-organ counting.

This repository provides **SPROUT-S/B/L checkpoints**, **pretraining code**, and **downstream training and inference code**. The **MCD-2.6M pretraining dataset** is available on [Hugging Face](https://huggingface.co/datasets/XIANG-Shuai/MCD-2.6m).

## Applications

SPROUT supports four plant phenotyping tasks:

| Task | Evaluated crops or settings |
|---|---|
| Plant organ segmentation | Apple, peach, pear, grape, wheat, and rice |
| Crop–weed parsing | Ten crop-field datasets |
| Monocular depth estimation | Sugar beet |
| Yield-organ counting | Wheat spikes and soybean pods |

## MCD-2.6M Pretraining Dataset

**MCD-2.6M** contains **2,604,450 open-field images** spanning crops, growth stages, sites, and field conditions. It is the image corpus used to pretrain SPROUT.

Download the images from [Hugging Face](https://huggingface.co/datasets/XIANG-Shuai/MCD-2.6m). The dataset card lists the sources, licenses, and attribution requirements.

## Installation

Use Python 3.8 for pretraining. For CUDA 11.6, install PyTorch and torchvision first:

```bash
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

Install the pretraining dependencies:

```bash
pip install -r requirements-pretraining.txt
```

For downstream tasks, follow the [DODA environment setup](https://github.com/illrayy/DODA). Run commands from the repository root.

## Pretrained Weights

Download a checkpoint and place it in `pretrained_models/`. Choose the matching S/B/L configuration when training a downstream model.

| Model | Params | Iterations | Download |
|---|---|---|---|
| SPROUT-S | 51M | 300k | [Google Drive](https://drive.google.com/file/d/1sRXmw1V2sxpmFR-Sy-ywl20NTNTB-ilf/view?usp=drive_link) |
| SPROUT-B | 112M | 450k | [Google Drive](https://drive.google.com/file/d/1GbFGf84psFy3-tSXKUx50eyquB5umK9b/view?usp=sharing) |
| SPROUT-L | 361M | 700k | [Google Drive](https://drive.google.com/file/d/1BVeLSpT13MdmhltrkaN5aYX3-yW9-1i1/view?usp=sharing) |

## Dataset Format

### Segmentation

```
your_dataset/
├── train/
│   ├── images/          # RGB images (e.g., 001.png, 002.png, ...)
│   └── masks/           # Segmentation masks (same filenames as images)
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

- Images and masks must share **identical filenames** (e.g., `001.png` in both `images/` and `masks/`).
- Training images must be **square** (height == width).
- Masks should contain integer class IDs starting from 0. Use `255` as the ignore label.

### Depth Estimation

```
your_dataset/
├── train/
│   ├── images/          # RGB images
│   └── depth/           # Depth maps (same filenames as images)
└── val/
    ├── images/
    └── depth/
```

- Images and depth maps must share **identical filenames**.
- Depth maps should be **16-bit PNG** files with values in **millimeters** (the code divides by 1000 to convert to meters).
- Depth values outside the valid range (`min_depth` to `max_depth`, in meters) are treated as invalid.

### Regression

```
your_dataset/
├── train/
│   ├── images/          # RGB images
│   └── train.txt        # Annotation file
├── val/
│   ├── images/
│   └── val.txt
└── test/
    ├── images/
    └── test.txt
```

List each image filename and its numeric target on one line, separated by a space:

```
image001.png 42
image002.png 15
image003.png 87
```

## Training

### Pretraining from unlabeled images

Create a UTF-8 text file with one image path per line:

```text
wheat/001.png
maize/002.jpg
```

Paths are absolute or relative to `--data-root`; spaces in filenames are supported. Pretraining uses RGB images without annotations.

Train SPROUT-S with a fresh output directory:

```bash
python pretrain.py --config config/sprout-S-pretrain.yaml --image-list datasets/train.txt --data-root datasets/images --output-dir logs/pretrain-S --gpus 1
```

Choose `sprout-B-pretrain.yaml` or `sprout-L-pretrain.yaml` for the other model sizes. The default input size is 256 × 256, with batch size 16 per GPU. Set training parameters in the YAML, or pass `--batch-size`, `--max-steps`, and `--accumulate-grad-batches`.

Effective batch size = per-GPU batch size × GPU count × accumulation.

For Linux multi-GPU training with FP16:

```bash
python pretrain.py --config config/sprout-L-pretrain.yaml --image-list datasets/train.txt --data-root datasets/images --output-dir logs/pretrain-L --gpus 4 --precision 16
```

The output directory contains:

| File | Contents |
|---|---|
| `pretraining.yaml` / `arguments.yaml` | Model and run settings |
| `metrics/` | CSV training metrics |
| `SPROUT-step=*.ckpt` | Checkpoints every 25,000 optimizer steps |
| `last.ckpt` | Final training checkpoint |

Checkpoints save model weights, EMA, diffusion buffers, the image normalization scale, optimizer, scheduler, and mixed-precision state when enabled.

Resume a run with its original config and output directory:

```bash
python pretrain.py --config config/sprout-S-pretrain.yaml --image-list datasets/train.txt --data-root datasets/images --output-dir logs/pretrain-S --resume logs/pretrain-S/last.ckpt --gpus 1
```

`--max-steps` sets the total target, including completed steps. Lightning 1.4 resumes at an epoch boundary and starts a new data order. Use checkpoints from `pretrain.py` for full-state resume; use the released pretrained weights to initialize downstream models.

To fine-tune your pretrained model, set `weight` to its checkpoint and select the matching S/B/L downstream config. Set the downstream script's `scale` to the value saved in your checkpoint:

```python
checkpoint = torch.load("logs/pretrain-S/last.ckpt", map_location="cpu")
scale = checkpoint["state_dict"]["scale_factor"].item()
```

### Downstream training

Set the dataset, checkpoint, model configuration, and training parameters in each script's `__main__` block:

| Variable | Description |
|---|---|
| `dataset_path` | Path to your dataset root directory |
| `num_classes` | Number of segmentation classes (segmentation only) |
| `max_number` | Maximum regression target value (regression only) |
| `min_depth` / `max_depth` | Valid depth range in meters (depth only) |
| `weight` | Path to pretrained checkpoint |
| `configs` | Path to model config YAML |
| `input_shape` | Input resolution (default: 256 for seg/depth, 384 for regression) |
| `time_step` | Diffusion timestep (default: 50 for seg/depth, 10 for regression) |
| `lr` | Learning rate |
| `batch_size` | Batch size |
| `total_iters` | Total training iterations |

### Train Segmentation

Set `dataset_path`, `num_classes`, `weight`, and `configs` in `train_segmentation.py`, then run:

```bash
python train_segmentation.py
```

- **Loss**: Focal Loss + Lovász Loss
- **LR schedule**: Cosine with warmup
- **Validation**: Sliding-window mIoU evaluation
- **Checkpoints**: Saved to `logs/segmentation/`

### Train Depth Estimation

Set the parameters in `train_depth_estimation.py`, then run:

```bash
python train_depth_estimation.py
```

- **Loss**: SiLog Loss
- **Validation**: RMSE and other depth metrics with horizontal flip TTA
- **Checkpoints**: Saved to `logs/depth/`

### Train Regression

Set the parameters in `train_regression.py`, then run:

```bash
python train_regression.py
```

- **Loss**: MSE Loss
- **Output mapping**: `sigmoid(output) * max_number`
- **Validation**: R², MAE, MSE, RMSE, MAPE
- **Checkpoints**: Saved to `logs/regression/`

## Inference

### Segmentation (Multi-Scale)

Set `dataset_path`, `weight`, `configs`, and `num_classes` in `segmentation_ms_inference.py`, then run:

```bash
python segmentation_ms_inference.py
```

- Uses multi-scale inference at scales `[0.75, 1.0, 1.25]` with sliding window (`window_size=256`, `step_size=128`).
- Applies horizontal flip TTA (test-time augmentation).
- Reports mIoU on the test set.

### Regression

Set `dataset_path`, `weight`, `configs`, and `max_number` in `regression_inference.py`, then run:

```bash
python regression_inference.py
```

- Applies horizontal flip TTA.
- Reports R², MAE, MSE, and RMSE on the test set.

### Depth Visualization

Set `weight`, `min_depth`, and `max_depth` in `visiualize_depth.py`, then run:

```bash
python visiualize_depth.py
```

- Place input images in `visualization/depth/input/`.
- Colorized depth maps (rainbow colormap) are saved to `visualization/depth/output/{model_name}/`.

## Citation

Cite SPROUT when using the models, code, or MCD-2.6M dataset:

```bibtex
@article{xiang2026sprout,
  title={SPROUT: A Scalable Diffusion Foundation Model for Multi-Crop Plant Phenotyping},
  author={Xiang, Shuai and Burridge, James and Liu, Shouyang and Lu, Hao and Fukatsu, Tokihiro and Zheng, Yinqiang and Guo, Wei},
  journal={arXiv preprint arXiv:2603.27519},
  year={2026},
  doi={10.48550/arXiv.2603.27519},
  url={https://arxiv.org/abs/2603.27519}
}
```
