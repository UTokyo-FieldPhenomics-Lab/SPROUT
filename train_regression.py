import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import seed_everything


import os
import cv2
import math
import random
import numpy as np
from functools import partial


from utils.utils import create_model, load_state_dict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error





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



class regressionDataset(Dataset):
    def __init__(self, img_path, anno_path, input_shape,scale=1):
        self.img_path = img_path
        self.input_shape = input_shape
        self.scale = scale
        f = open(anno_path, 'r').readlines()

        labels = {}
        for line in f:
            line = line.split()
            img_name = line[0]
            num_gt = int(line[1])
            labels[img_name] = num_gt
        self.labels = labels
        self.img_names = list(self.labels.keys())


    def __len__(self):
        return len(self.img_names)
        
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.img_path, img_name)), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_shape, self.input_shape))

        if random.random() < 0.5:
            img = cv2.flip(img, 1)

        img = (img.astype(np.float32)/127.5-1) * self.scale

        return img, self.labels[img_name]

    




if __name__ == '__main__':
    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_number = 120


    warmup_iters = 100
    total_iters = 500
    val_step = 20
    batch_size = 4
    accumulation_steps = 4
    input_shape = 384
    time_step = 10

    scale = 2.0209 # set the scale according to the pretrained model

    dataset_path = '' # change to your dataset path

    lr = 5e-5

    configs = 'configs/sprout-L-regression.yaml'
    weight = "pretrained_models/SPORUT-L_step=700k.ckpt" # download the pretrained weight from google drive

    ckpt_name = weight.split('/')[-1].split('.')[0] + f'_ts{time_step}_lr{lr}'


    model = create_model(configs).cpu()
    pretrained_weight = load_state_dict(weight, location=device)
    unet_weight = {k.replace('model.diffusion_model.', ''):v for k,v in pretrained_weight.items() if 'model.diffusion_model.' in k and 'out.2' not in k and 'out.0' not in k}
    model.load_state_dict(unet_weight, strict=False)
    model = model.to(device)



    optimizer = optim.AdamW(model.parameters(), 0, weight_decay = 5e-4)

    train_set = regressionDataset(f'{dataset_path}/train/images', f'{dataset_path}/train/train.txt', input_shape=input_shape, scale=scale)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, persistent_workers=True, num_workers=4, pin_memory=True, drop_last=True)
    val_set = regressionDataset(f'{dataset_path}/val/images', f'{dataset_path}/val/val.txt', input_shape=input_shape, scale=scale)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, persistent_workers=True, num_workers=2)

    

    os.makedirs(f'logs/regression', exist_ok=True)



    epoch_step = len(train_set) // (batch_size*accumulation_steps)
    num_epochs = math.ceil(total_iters / epoch_step)
    lr_scheduler_func = get_lr_scheduler(lr=lr, warmup_total_iters=warmup_iters, total_iters=total_iters)

    criterion_mse = nn.MSELoss(reduction="sum")

    running_loss = 0.0
    iters_now = 0
    best_mse = float('inf')
    for epoch in range(num_epochs):
        print(f'start the training of epoch {epoch}')

        for batch_idx, (imgs, num_gt) in enumerate(train_loader):
            
            imgs, num_gt = imgs.to(device),  num_gt.to(device).float()
            t = torch.full((imgs.shape[0],), time_step, device=device).long()
            imgs = imgs.permute(0, 3, 1, 2)

            # 前向传播
            outputs = model(imgs, t)
            outputs = torch.sigmoid(outputs) * max_number

            # 计算损失
            loss = criterion_mse(outputs, num_gt.unsqueeze(1))
            running_loss += loss.item()
            loss = loss / accumulation_steps
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

                    groud_truth = []
                    predict_result = []
                    for imgs, num_gt in val_loader:
                        imgs = imgs.permute(0, 3, 1, 2).to(device)
                        t = torch.full((imgs.shape[0],), time_step, device=device).long()

                        with torch.no_grad():
                            outputs = model(imgs, t).cpu()
                            outputs = torch.sigmoid(outputs) * max_number

                        groud_truth.append(num_gt.item())
                        predict_result.append(float(outputs.mean().item()))

                    predict_result = np.array(predict_result)
                    groud_truth = np.array(groud_truth)
                    r2 = r2_score(predict_result, groud_truth)
                    mae = mean_absolute_error(predict_result, groud_truth)
                    mse = mean_squared_error(predict_result, groud_truth)
                    mape = ((np.abs(predict_result - groud_truth) / groud_truth) * 100).mean()
                    rmse = np.sqrt(mse)

                    if mse <= best_mse:
                        best_mse = mse
                        print(f"mse: {best_mse:.4f}, r2: {r2:.4f}, mae: {mae:.4f}, rmse: {rmse:.4f}, mape: {mape:.4f}")
                        torch.save(model.state_dict(), f'logs/regression/{ckpt_name}.ckpt')
                    
                    model.train()
                if iters_now >= total_iters:
                    break

            
