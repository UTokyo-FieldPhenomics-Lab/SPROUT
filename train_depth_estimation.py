import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import seed_everything


import os
import cv2
import math
import random
import numpy as np
from functools import partial



from utils.utils import create_model, load_state_dict
from sprout.depth_model import FeaturesToDepth
from utils.depth_metrics import calculate_depth_metrics



class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target):
        valid_mask = (target > 0).detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss



def get_lr_scheduler(lr, total_iters, warmup_total_iters, warmup_lr_start = 0):
    func = partial(warm_cos_lr, lr, total_iters, warmup_total_iters, warmup_lr_start,)
    return func

def warm_cos_lr(lr, total_iters, warmup_total_iters, warmup_lr_start, iters):
    """Cosine learning rate with warm up."""
    if iters <= warmup_total_iters:
        lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
    else:
        lr *= 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (iters - warmup_total_iters)
                / (total_iters - warmup_total_iters)
            )
        )
    return lr

def set_optimizer_lr(optimizer, lr_scheduler_func, iter):
    lr = lr_scheduler_func(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



class depthDataset(Dataset):
    def __init__(self, img_path, depth_path, input_shape, ratio_range=(0.5, 2), base_size=256, scale=1, is_train=True, min_depth=0.9, max_depth=1.2):
        self.depth_path = []
        self.img_path = []
        self.input_shape = input_shape
        self.ratio_range = ratio_range
        self.base_size = base_size
        self.scale = scale
        self.is_train = is_train
        self.min_depth = min_depth
        self.max_depth = max_depth
        for img_name in os.listdir(img_path):
            self.img_path.append(os.path.join(img_path, img_name))
            self.depth_path.append(os.path.join(depth_path, img_name))
        
    def __len__(self):
        return len(self.img_path)
    

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.img_path[idx]), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(self.depth_path[idx], cv2.IMREAD_UNCHANGED) / 1000
        depth[depth < self.min_depth] = -1
        depth[depth > self.max_depth] = -1

        if not self.is_train:
            img = (img.astype(np.float32)/127.5-1) * self.scale
            img = cv2.resize(img, (self.input_shape, self.input_shape))
            depth = cv2.resize(depth, (self.input_shape, self.input_shape), interpolation=cv2.INTER_NEAREST)
            return img, depth

        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            depth = cv2.flip(depth, 1)
        
        img = cv2.resize(img, (self.input_shape, self.input_shape))
        depth = cv2.resize(depth, (self.input_shape, self.input_shape), interpolation=cv2.INTER_NEAREST)

        img = (img.astype(np.float32)/127.5-1) * self.scale
        return img, depth
    

class depthModel(nn.Module):
    def __init__(self, model, max_depth, min_depth):
        super().__init__()
        self.encoder = model
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




if __name__ == '__main__':
    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_depth = 1.2 # The unit is meters.
    min_depth = 0.9

    
    warmup_iters = 100
    total_iters = 500
    val_step = 50
    batch_size = 4
    accumulation_steps = 4
    input_shape = 256
    time_step = 50

    scale = 2.0209 # set the scale according to the pretrained model

    dataset_path = '' # change to your dataset path

    lr = 2e-5

    configs = 'configs/sprout-L-depth.yaml'
    weight = "pretrained_models/SPORUT-L_step=700k.ckpt" # download the pretrained weight from google drive

    ckpt_name = weight.split('/')[-1].split('.')[0] + f'_ts{time_step}_lr{lr}'


    encoder = create_model(configs).cpu()
    pretrained_weight = load_state_dict(weight, location=device)
    unet_weight = {k.replace('model.diffusion_model.', ''):v for k,v in pretrained_weight.items() if 'model.diffusion_model.' in k}
    encoder.load_state_dict(unet_weight, strict=False)

    model = depthModel(encoder, max_depth, min_depth).to(device)



    optimizer = optim.AdamW(model.parameters(), 0)

    train_set = depthDataset(f'{dataset_path}/train/images', f'{dataset_path}/train/depth', input_shape=input_shape, scale=scale, is_train=True, min_depth=min_depth, max_depth=max_depth)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, persistent_workers=True, num_workers=4, pin_memory=True, drop_last=True)
    val_set = depthDataset(f'{dataset_path}/val/images', f'{dataset_path}/val/depth', input_shape=input_shape, scale=scale, is_train=False, min_depth=min_depth, max_depth=max_depth)
    val_loader = DataLoader(val_set, batch_size=batch_size//2, shuffle=False, persistent_workers=True, num_workers=2)
    

    os.makedirs(f'logs/depth', exist_ok=True)



    epoch_step = len(train_set) // (batch_size*accumulation_steps)
    num_epochs = math.ceil(total_iters / epoch_step)
    lr_scheduler_func = get_lr_scheduler(lr=lr, warmup_total_iters=warmup_iters, total_iters=total_iters)

    running_loss = 0.0

    criterion = SiLogLoss()

    iters_now = 0
    best_rmse = float('inf')
    for epoch in range(num_epochs):
        print(f'start the training of epoch {epoch}')

        for batch_idx, (imgs, depths) in enumerate(train_loader):
            
            imgs, depths = imgs.to(device),  depths.to(device)
            imgs = imgs.permute(0, 3, 1, 2)
            t = torch.full((imgs.shape[0],), time_step, device=device).long()

            # 前向传播
            outputs = model(imgs, t).squeeze(1)
            loss = criterion(outputs, depths)
            loss = loss / accumulation_steps
            

            # 计算损失
            # 记录损失
            running_loss += loss.item()

            loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                set_optimizer_lr(optimizer, lr_scheduler_func, iters_now)
                iters_now += 1
        
                if iters_now % val_step == 0:
                    model.eval()
                    print(f"Iters {iters_now}, Train Loss: {running_loss/val_step}")
                    running_loss = 0.0


                    pred_depths = []
                    gt_depths = []
                    for imgs, depths in val_loader:
                        imgs = imgs.to(device)
                        imgs = imgs.permute(0, 3, 1, 2)
                        t = torch.full((imgs.shape[0],), time_step, device=device).long()
                        
                        with torch.no_grad():
                            outputs = model(imgs, t, flip=True).cpu().squeeze(1)

                        pred_depths.append(outputs.cpu())
                        gt_depths.append(depths)
                    pred_depths = torch.cat(pred_depths, dim=0)
                    gt_depths = torch.cat(gt_depths, dim=0)
                    metrics = calculate_depth_metrics(gt_depths, pred_depths)
                    rmse = float(metrics['rmse'].item())

                    
                    if rmse < best_rmse:
                        best_rmse = rmse
                        print(f"Best RMSE: {best_rmse:.4f}")
                        print(metrics)
                        torch.save(model.state_dict(), f'logs/depth/{ckpt_name}.ckpt')
                    
                    model.train()
            
