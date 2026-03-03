import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.utils import create_model, load_state_dict
from utils.evaluate import fast_hist, per_class_iou
import torch
import torch.nn.functional as F




class msDataset(Dataset):
    def __init__(self, img_path, mask_path, input_shape, num_classes, scales=[0.75, 1, 1.25], scale=1, scale_weights=None):
        self.mask_path = []
        self.img_path = []
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.scales = scales

        self.scale = scale
        # If no weights are provided, use equal weights
        self.scale_weights = scale_weights if scale_weights is not None else [1.0] * len(scales)
        for mask_name in os.listdir(mask_path):
            self.mask_path.append(os.path.join(mask_path, mask_name))
            self.img_path.append(os.path.join(img_path, mask_name))
        
    def __len__(self):
        return len(self.mask_path)
        
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.img_path[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path[idx])

        imgs = []

        for scale_factor in self.scales:
            # First scale, then normalize
            scaled_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
            # Unified normalization method to avoid applying scale repeatedly
            normalized_img = (scaled_img.astype(np.float32)/127.5-1) * self.scale
            imgs.append(normalized_img)


        return imgs, mask
    
    
if __name__ == '__main__':
    dataset_path = '' # change to your dataset path
    configs = 'configs/sprout-L-seg.yaml'

    weight = 'logs/segmentation/SPROUT-L_step=700k_ts50_lr2e-05.ckpt' # change to your model weight

    img_folder = f'{dataset_path}/test/images'
    mask_folder = f'{dataset_path}/test/masks'


    scale = 2.0209 # set the scale according to the pretrained model
    num_classes = 4

    window_size = 256
    step_size = 128

    


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = create_model(configs).cpu()

    model.load_state_dict(load_state_dict(weight, location=device))
    model = model.to(device)

    model.eval()

    dataset = msDataset(img_folder, mask_folder, input_shape=256, num_classes=num_classes, scale=scale)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, persistent_workers=False, num_workers=2)

    hist = np.zeros((num_classes, num_classes))
    for imgs, mask in tqdm(test_loader, desc='Processing images'):
        mask = mask.cpu().numpy()
        b, h_mask, w_mask = mask.shape

        ms_result = torch.zeros((1, num_classes, h_mask, w_mask), dtype=torch.float32, device='cpu')
        total_weight = 0  # Used to compute the weighted average

        t = torch.full((2,), 50, device=device).long()
        for scale_idx, img in enumerate(imgs):
            _, h, w, _ = img.shape
            img = img.permute(0, 3, 1, 2).cuda().float()
            result = torch.zeros((1, num_classes, h, w), dtype=torch.float32, device='cpu')
            count_map = torch.zeros_like(result)
            for y in range(0, h, step_size):
                for x in range(0, w, step_size):
                    y_end = min(y + window_size, h)
                    x_end = min(x + window_size, w)
                    
                    # If the window is too small, adjust the starting position
                    if y_end - y < window_size:
                        y = max(0, h - window_size)
                        y_end = h
                    if x_end - x < window_size:
                        x = max(0, w - window_size)
                        x_end = w
                    
                    # Extract window
                    window = img[:, :, y:y_end, x:x_end]
                    valid_h = window.shape[2]
                    valid_w = window.shape[3]
                    
                    if valid_h < window_size or valid_w < window_size:
                        padded_window = torch.zeros((1, 3, window_size, window_size), dtype=torch.float32, device=device)
                        padded_window[:, :, :valid_h, :valid_w] = window
                        window = padded_window


                    window_flip = torch.flip(window, [3])
                    window = torch.cat([window, window_flip], dim=0)

                    with torch.no_grad():
                        pred = model(window, t)
                        pred = (pred[0] + torch.flip(pred[1], [2])) / 2
                        pred = pred.unsqueeze(0).cpu()
                        
                    # Only take the prediction results of the valid region
                    valid_predict = pred[:, :, :valid_h, :valid_w]  # (1, num_classes, valid_h, valid_w)
                    
                    # Accumulate results
                    result[:, :, y:y+valid_h, x:x+valid_w] += valid_predict
                    count_map[:, :, y:y+valid_h, x:x+valid_w] += 1

            result = result / count_map
            # Interpolate to the original mask size and accumulate with weights
            interpolated_result = F.interpolate(result, size=(h_mask, w_mask), mode='bilinear', align_corners=False)
            weight = dataset.scale_weights[scale_idx]
            ms_result += interpolated_result * weight
            total_weight += weight
            
        # Perform weighted averaging on multi-scale results
        ms_result = ms_result / total_weight
        final = ms_result.permute(0,2,3,1).cpu().numpy().argmax(axis=-1)


        hist += fast_hist(mask.flatten(), final.flatten(), num_classes)
    miou = np.mean(per_class_iou(hist))
    print(f'miou: {miou}')
