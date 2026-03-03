import torch
import numpy as np
import torch.nn.functional as F


def fast_hist(a, b, n):

    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n) 

def per_class_iou(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

def evaluate_crop(model, time_step, loader, num_classes=19):
    hist = np.zeros((num_classes, num_classes))

    for batch_idx, (imgs, ref_imgs, masks) in enumerate(loader):
        imgs = imgs.permute(0, 3, 1, 2).cuda().float()
        ref_imgs = ref_imgs.cuda()
        masks = masks.cpu().numpy()

        b, h, w = masks.shape
        b, _, h_i, w_i = imgs.shape
        final = torch.zeros(b, num_classes, h, w).cuda()
        t = torch.full((b,), time_step).cuda().long()
        
        with torch.no_grad():
            pred = model(imgs[:, :, : , :h_i], t, context=ref_imgs)   # w = 2h
            final[:, :, : , :h] += pred.softmax(dim=1)

            pred = model(imgs[:, :, : , h_i:], t, context=ref_imgs)
            final[:, :, : , h:] += pred.softmax(dim=1)

            final = final.argmax(dim=1).cpu().numpy()
        if b > 1:
            for i in range(b):
                hist += fast_hist(masks[i].flatten(), final[i].flatten(), num_classes)
    miou = np.mean(per_class_iou(hist))
    return miou

def evaluate(model, time_step, loader, num_classes=19, cond_model=None):
    hist = np.zeros((num_classes, num_classes))

    for batch_idx, batch in enumerate(loader):
        if cond_model is not None:
            imgs, masks, cond = batch
        else:
            imgs, masks = batch

        imgs = imgs.permute(0, 3, 1, 2).cuda().float()
        masks = masks.cpu().numpy()

        b, h, w = masks.shape
        t = torch.full((b,), time_step).cuda().long()
        
        with torch.no_grad():
            if cond_model is not None:
                cond = cond_model(cond.cuda()).last_hidden_state[:, 0, :].unsqueeze(1)
                pred = model(imgs, t, context=cond)
            else:
                pred = model(imgs, t)
                pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=False)
            final = F.softmax(pred.permute(0,2,3,1),dim = -1).cpu().numpy().argmax(axis=-1)

        if b >= 1:
            for i in range(b):
                hist += fast_hist(masks[i].flatten(), final[i].flatten(), num_classes)
    miou = np.mean(per_class_iou(hist))
    return miou

def evaluate_slides(model, time_step=None, loader=None, num_classes=19, window_size=256, step_size=256, context=None):
    hist = np.zeros((num_classes, num_classes))

    for batch_idx, batch in enumerate(loader):
        imgs, masks = batch
        imgs = imgs.permute(0, 3, 1, 2).cuda().float()
        masks = masks.cpu().numpy()

        b, h, w = masks.shape
        result = torch.zeros((b, num_classes, h, w), dtype=torch.float32).cuda()
        count_map = torch.zeros_like(result)

        t = torch.full((b,), time_step).cuda().long()
        for y in range(0, h, step_size):
            for x in range(0, w, step_size):
                y_end = min(y + window_size, h)
                x_end = min(x + window_size, w)
                
                # 如果窗口太小，调整起始位置
                if y_end - y < window_size:
                    y = max(0, h - window_size)
                    y_end = h
                if x_end - x < window_size:
                    x = max(0, w - window_size)
                    x_end = w
                
                # 提取窗口
                window = imgs[:, :, y:y_end, x:x_end]
                valid_h = window.shape[2]
                valid_w = window.shape[3]
                
                # 如果窗口尺寸不足，进行padding
                if valid_h < window_size or valid_w < window_size:
                    padded_window = torch.zeros((b, 3, window_size, window_size), dtype=torch.float32)
                    padded_window[:, :, :valid_h, :valid_w] = window
                    window = padded_window
                
                with torch.no_grad():
                    if context is not None:
                        pred = model(window, t, context=context.clone().repeat(window.shape[0], 1, 1))
                    else:
                        pred = model(window, t)

                # 只取有效区域的预测结果
                valid_predict = pred[:, :, :valid_h, :valid_w]  # (1, num_classes, valid_h, valid_w)
                
                # 累加结果
                result[:, :, y:y+valid_h, x:x+valid_w] += valid_predict
                count_map[:, :, y:y+valid_h, x:x+valid_w] += 1

        count_map[count_map == 0] = 1  # 避免除零
        result = result / count_map

        final = F.softmax(result.permute(0,2,3,1),dim = -1).cpu().numpy().argmax(axis=-1)

        for i in range(b):
            hist += fast_hist(masks[i].flatten(), final[i].flatten(), num_classes)
    ious = per_class_iou(hist)
    print(np.round(ious*100, 2))
    
    return np.mean(ious)

