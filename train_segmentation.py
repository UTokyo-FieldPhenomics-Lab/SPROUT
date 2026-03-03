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

from utils.lovasz_loss import LovaszLoss

from utils.utils import load_state_dict, instantiate_from_config
from utils.evaluate import evaluate_slides
from omegaconf import OmegaConf



def CE_Loss(inputs, target, cls_weights=None):
    _, c, h, w = inputs.size()
    _, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=False)

    loss = F.cross_entropy(
        inputs,
        target,
        weight=cls_weights,
        reduction='none',
        ignore_index=255)
    
    avg_factor = target.numel()
    loss = loss.sum() / avg_factor
    return loss

def Focal_Loss(inputs, target, cls_weights=None, num_classes=21, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt  = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=255, reduction='none')(temp_inputs, temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss

def Dice_loss(inputs, target, beta=1, smooth = 1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   计算dice loss
    #--------------------------------------------#
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss



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



class segDataset(Dataset):
    def __init__(self, img_path, mask_path, input_shape, num_classes, ratio_range=(0.5, 2), base_size=256, scale=1, cat_max_ratio=0.75, is_train=True):
        self.mask_path = []
        self.img_path = []
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.ratio_range = ratio_range
        self.base_size = base_size
        self.scale = scale
        self.is_train = is_train
        self.cat_max_ratio = cat_max_ratio
        for img_name in os.listdir(img_path):
            self.img_path.append(os.path.join(img_path, img_name))
            self.mask_path.append(os.path.join(mask_path, img_name))
        
    def __len__(self):
        return len(self.img_path)
    
    def generate_crop_bbox(self, img):
        margin_h = max(img.shape[0] - self.input_shape, 0)
        margin_w = max(img.shape[1] - self.input_shape, 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.input_shape
        crop_x1, crop_x2 = offset_w, offset_w + self.input_shape
        return crop_y1, crop_y2, crop_x1, crop_x2
    
    def random_crop(self, img, mask):
        mask_copy = mask.copy()

        crop_bbox = self.generate_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = mask_copy[crop_bbox[0]:crop_bbox[1], crop_bbox[2]:crop_bbox[3]]
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != 255]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.generate_crop_bbox(img)
        img = img[crop_bbox[0]:crop_bbox[1], crop_bbox[2]:crop_bbox[3]]
        mask = mask[crop_bbox[0]:crop_bbox[1], crop_bbox[2]:crop_bbox[3]]
        return img, mask
        
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.img_path[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path[idx])

        if not self.is_train:
            img = (img.astype(np.float32)/127.5-1) * self.scale
            return img, mask

        h, w = img.shape[:2]
        assert h == w, "The height and width of the img must be the same"
        ratio = np.random.uniform(*self.ratio_range)

        img = cv2.resize(img, None, fx=ratio, fy=ratio)
        mask = cv2.resize(mask, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)

        img, mask = self.random_crop(img, mask)

        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        img = (img.astype(np.float32)/127.5-1) * self.scale
        # pad to input_shape
        img = cv2.copyMakeBorder(img, 0, self.input_shape - img.shape[0], 0, self.input_shape - img.shape[1], cv2.BORDER_CONSTANT, value=0)
        mask = cv2.copyMakeBorder(mask, 0, self.input_shape - mask.shape[0], 0, self.input_shape - mask.shape[1], cv2.BORDER_CONSTANT, value=255)

        return img, mask
    

    




if __name__ == '__main__':
    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 4 # change to the number of classes in your dataset
    
    warmup_iters = 100
    total_iters = 500
    val_step = 10
    batch_size = 4
    accumulation_steps = 4
    input_shape = 256
    time_step = 50

    scale = 2.0209 # set the scale according to the pretrained model

    dataset_path = '' # change to your dataset path
    lovasz_loss = LovaszLoss(reduction='none', loss_weight=1)

    

    lr = 2e-5

    configs = 'configs/sprout-L-seg.yaml' 
    weight = "pretrained_models/SPORUT-L_step=700k.ckpt" # download the pretrained weight from google drive

    ckpt_name = weight.split('/')[-1].split('.')[0] + f'_ts{time_step}_lr{lr}'




    config = OmegaConf.load(configs)
    config.model.params.num_classes = num_classes
    model = instantiate_from_config(config.model).cpu()

    pretrained_weight = load_state_dict(weight, location=device)
    unet_weight = {k.replace('model.diffusion_model.', ''):v for k,v in pretrained_weight.items() if 'model.diffusion_model.' in k and 'out.2' not in k}
    model.load_state_dict(unet_weight, strict=False)
    model = model.to(device)



    optimizer = optim.AdamW(model.parameters(), 0)

    train_set = segDataset(f'{dataset_path}/train/images', f'{dataset_path}/train/masks', input_shape=input_shape, num_classes=num_classes, scale=scale, is_train=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, persistent_workers=True, num_workers=4, pin_memory=True, drop_last=True)
    val_set = segDataset(f'{dataset_path}/val/images', f'{dataset_path}/val/masks', input_shape=input_shape, num_classes=num_classes, scale=scale, is_train=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, persistent_workers=False, num_workers=2)
    
    
    

    os.makedirs(f'logs/segmentation', exist_ok=True)



    epoch_step = len(train_set) // (batch_size*accumulation_steps)
    num_epochs = math.ceil(total_iters / epoch_step)
    lr_scheduler_func = get_lr_scheduler(lr=lr, warmup_total_iters=warmup_iters, total_iters=total_iters)

    running_loss = 0.0

    iters_now = 0
    best_miou = 0.0
    for epoch in range(num_epochs):
        print(f'start the training of epoch {epoch}')

        for batch_idx, (imgs, masks) in enumerate(train_loader):
            
            imgs, masks = imgs.to(device),  masks.to(device).long()
            t = torch.full((imgs.shape[0],), time_step, device=device).long()
            imgs = imgs.permute(0, 3, 1, 2)


            outputs = model(imgs, t)


            loss = Focal_Loss(outputs, masks) + lovasz_loss(outputs, masks)
            loss = loss / accumulation_steps

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

                    miou = evaluate_slides(model, time_step, val_loader, num_classes=num_classes, window_size=input_shape, step_size=input_shape)
                    
                    if miou >= best_miou:
                        best_miou = max(miou, best_miou)
                        print(f"Best mIoU: {best_miou:.4f}")
                        torch.save(model.state_dict(), f'logs/segmentation/{ckpt_name}.ckpt')
                    
                    model.train()
            
