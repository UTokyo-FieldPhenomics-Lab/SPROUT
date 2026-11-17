# SPROUT

SPROUT is a multi-crop, multi-task agricultural foundation model trained via diffusion denoising.

This repository currently supports three types of downstream tasks:
- **Semantic Segmentation** 
- **Monocular Depth Estimation** 
- **Counting** 



### Dependencies

Refer to https://github.com/illrayy/DODA to set up the environment.

### Pretrained Weights

Download the pretrained checkpoint and place it under `pretrained_models/`.

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

Annotation file format (space-separated, one sample per line):

```
image001.png 42
image002.png 15
image003.png 87
```

Each line contains the image filename and its corresponding numeric label.

## Training

### Pretraining from unlabeled images

Install the pretraining runtime (Python 3.8, PyTorch 1.13.0 and Lightning 1.4.2):

```bash
pip install -r requirements-pretraining.txt
```

For CUDA 11.6, install the matching PyTorch wheels first:

```bash
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

Prepare a UTF-8 text file with **one RGB image path per line**; annotations are not required. Paths may be absolute or relative to `--data-root`, and may contain spaces. For example:

```text
wheat/001.png
maize/002.jpg
```

Run from the repository root, using a fresh output directory:

```bash
python pretrain.py --config config/sprout-S-pretrain.yaml --image-list datasets/train.txt --data-root datasets/images --output-dir logs/pretrain-S --gpus 1
```

Use `sprout-B-pretrain.yaml` or `sprout-L-pretrain.yaml` for the other model sizes. Defaults are 256 × 256 images, batch size 16 **per GPU**, AdamW at 3.5e-5 (weight decay 0.01), 10,000 warmup optimizer steps, and 300k/450k/700k optimizer steps for S/B/L. These are runnable defaults adapted from the DPS crop trainer and the released checkpoint iteration counts; they are not a verified recipe for reproducing the released weights. Adjust `training` in the YAML or override `--batch-size`, `--max-steps`, and `--accumulate-grad-batches`. Effective batch size is per-GPU batch size × GPU count × accumulation.

The implementation follows the DPS pixel-space diffusion objective: normalize RGB to [-1, 1], multiply by the inverse standard deviation of the first training batch, sample one of 1,000 timesteps, and predict Gaussian noise with MSE. Betas follow the original square-root-linear schedule from 0.0002 to 0.02. The model computes its own class-token context; no VAE, DINO, text encoder, labels, or additional pretrained model downloads are needed. EMA uses the original adaptive decay capped at 0.9999 and updates after each training batch. Augmentation uses rotation with a contained square crop (probability 0.5), resize, and vertical flip (probability 0.5); the crop implementation is cleaned up from DPS and is not pixel-identical to its original augmentation.

For Linux multi-GPU training with FP16, for example:

```bash
python pretrain.py --config config/sprout-L-pretrain.yaml --image-list datasets/train.txt --data-root datasets/images --output-dir logs/pretrain-L --gpus 4 --precision 16
```

Use a smaller batch if GPU memory is insufficient. For a quick smoke run, add `--max-steps 2 --batch-size 1 --num-workers 0`; CPU runs also require `--gpus 0 --precision 32`.

The run directory contains `pretraining.yaml`, `arguments.yaml`, CSV metrics, periodic `SPROUT-step=*.ckpt` files (every 25,000 optimizer steps), and `last.ckpt` at completion. Checkpoints include model weights, EMA, diffusion buffers, `scale_factor`, optimizer, scheduler, and mixed-precision state when applicable. Resume a checkpoint produced by this entry point with the same config and run directory:

```bash
python pretrain.py --config config/sprout-S-pretrain.yaml --image-list datasets/train.txt --data-root datasets/images --output-dir logs/pretrain-S --resume logs/pretrain-S/last.ckpt --gpus 1
```

`--max-steps` is the **total** target, including steps already completed. Lightning 1.4 resumes at an epoch boundary; the exact mid-epoch image order and random augmentations are not restored. Legacy DPS/released weight-only checkpoints are for downstream initialization, not full-state resume through this entry point.

For downstream training, set `weight` to the new checkpoint and choose the matching S/B/L downstream config. The checkpoint preserves the `model.diffusion_model.*` keys used by the existing loaders. Also replace the downstream script's hardcoded `scale = 2.0209` with the value saved by your run:

```python
checkpoint = torch.load("logs/pretrain-S/last.ckpt", map_location="cpu")
scale = checkpoint["state_dict"]["scale_factor"].item()
```

Existing downstream loaders select the regular model weights; EMA buffers are retained for training continuity. This release focuses on pretraining and downstream transfer; it does not include diffusion image sampling or generation-quality validation. The original pretraining corpus and full-scale reproduction results are not bundled.

### Downstream training settings

The downstream training scripts use hardcoded hyperparameters in their `__main__` block. Before running, open the script and modify the following variables:

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

Edit `train_segmentation.py` and set your parameters, then run:

```bash
python train_segmentation.py
```

- **Loss**: Focal Loss + Lovász Loss
- **LR schedule**: Cosine with warmup
- **Validation**: Sliding-window mIoU evaluation
- **Checkpoints**: Saved to `logs/segmentation/`

### Train Depth Estimation

Edit `train_depth_estimation.py` and set your parameters, then run:

```bash
python train_depth_estimation.py
```

- **Loss**: SiLog Loss
- **Validation**: RMSE and other depth metrics with horizontal flip TTA
- **Checkpoints**: Saved to `logs/depth/`

### Train Regression

Edit `train_regression.py` and set your parameters, then run:

```bash
python train_regression.py
```

- **Loss**: MSE Loss
- **Output mapping**: `sigmoid(output) * max_number`
- **Validation**: R², MAE, MSE, RMSE, MAPE
- **Checkpoints**: Saved to `logs/regression/`

## Inference

### Segmentation (Multi-Scale)

Edit `segmentation_ms_inference.py` to set `dataset_path`, `weight`, `num_classes`, etc., then run:

```bash
python segmentation_ms_inference.py
```

- Uses multi-scale inference at scales `[0.75, 1.0, 1.25]` with sliding window (`window_size=256`, `step_size=128`).
- Applies horizontal flip TTA (test-time augmentation).
- Reports mIoU on the test set.

### Regression

Edit `regression_inference.py` to set `dataset_path`, `weight`, `max_number`, etc., then run:

```bash
python regression_inference.py
```

- Applies horizontal flip TTA.
- Reports R², MAE, MSE, and RMSE on the test set.

### Depth Visualization

Edit `visiualize_depth.py` to set `weight`, `min_depth`, `max_depth`, etc., then run:

```bash
python visiualize_depth.py
```

- Place input images in `visualization/depth/input/`.
- Colorized depth maps (rainbow colormap) are saved to `visualization/depth/output/{model_name}/`.