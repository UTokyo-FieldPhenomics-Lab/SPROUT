import torch
from pytorch_lightning import seed_everything
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


from utils.utils import create_model, load_state_dict

    




if __name__ == '__main__':
    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_number = 120

    input_shape = 384
    time_step = 50

    scale = 2.0209 # set the scale according to the pretrained model

    dataset_path = '' # change to your dataset path


    configs = 'configs/sprout-L-regression.yaml'
    weight = 'logs/regression/SPROUT-L_step=700k_ts50_lr5e-05.ckpt' # change to your model weight

    anns = open(f'{dataset_path}/test/test.txt', 'r').readlines()
    img_path = f'{dataset_path}/test/images'




    model = create_model(configs).cpu()
    model.load_state_dict(load_state_dict(weight, location=device), strict=False)
    model = model.to(device)

    t = torch.full((1,), time_step, device=device).long()

    groud_truth = []
    predict_result = []
    for ann in tqdm(anns):
        img_name, num_gt = ann.split()[0], ann.split()[1]
        groud_truth.append(int(num_gt))

        img = cv2.imread(f'{img_path}/{img_name}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (input_shape, input_shape)) /127.5 - 1
        img *= scale
        img_flip = cv2.flip(img, 1)
        img = np.array([img, img_flip])

        img = torch.from_numpy(img).to(device).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            outputs = model(img, t).cpu()
            outputs = torch.sigmoid(outputs) * max_number
        predict_result.append(float(outputs.mean().item()))


    predict_result = np.array(predict_result)
    groud_truth = np.array(groud_truth)

    r2 = r2_score(predict_result, groud_truth)
    mae = mean_absolute_error(predict_result, groud_truth)
    mse = mean_squared_error(predict_result, groud_truth)
    rmse = np.sqrt(mse)

    print(f"R^2: {r2}")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")


            
