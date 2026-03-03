import os
import cv2
import numpy as np
import torch
from utils.utils import create_model
from sprout.depth_model import FeaturesToDepth
import torch.nn as nn
import matplotlib


class depthModel(nn.Module):
    def __init__(self, encoder, max_depth, min_depth):
        super().__init__()
        self.encoder = encoder
        self.decoder = FeaturesToDepth(min_depth=min_depth, max_depth=max_depth)

    def forward(self, x, t, flip=False):
        if flip:
            x_flip = torch.flip(x, [-1])
            x_flip = self.encoder(x_flip, t)
            x_flip = torch.flip(x_flip, [-1])
        x = self.encoder(x, t)
        if flip:
            x = (x + x_flip)/2
        x = self.decoder(x)
        return x

min_depth = 0.9
max_depth = 1.2

scale = 2.0209 # set the scale according to the pretrained model

input_shape = 256

weight = 'logs/depth/SPROUT-L_step=700k_ts50_lr2e-05.ckpt' # change to your model weight

configs = 'configs/sprout-L-depth.yaml'

model_name = 'SPROUT-L'
encoder = create_model(configs).cpu()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = depthModel(encoder, max_depth, min_depth).to(device)
model.load_state_dict(torch.load(weight))

t = torch.full((1,), 50, device=device).long()

img_path = 'visualization/depth/input'
output_path = f'visualization/depth/output/{model_name}'
os.makedirs(output_path, exist_ok=True)

rainbow_cmap = matplotlib.colormaps['rainbow']

for img_name in os.listdir(img_path):
    img = cv2.imread(os.path.join(img_path, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_shape, input_shape))
    img = (img.astype(np.float32)/127.5-1) * scale
    img = torch.from_numpy(img).to(device).permute(2, 0, 1).float().unsqueeze(0)
    with torch.no_grad():
        depth = model(img, t, flip=True).cpu().squeeze().numpy()

    
    depth_norm = (depth - min_depth) / (max_depth - min_depth)
    depth_norm = 1-depth_norm
    depth_color = rainbow_cmap(depth_norm)
    
    # Convert to BGR format (OpenCV format) and scale to 0-255
    depth_color = (depth_color[:, :, :3] * 255).astype(np.uint8)
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_path, img_name), depth_color)