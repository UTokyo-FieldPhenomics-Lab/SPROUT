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

All training scripts use hardcoded hyperparameters in their `__main__` block. Before running, open the script and modify the following variables:

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