import os
import csv
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import networkx as nx
from omegaconf import OmegaConf
import importlib

import torch
import pytorch_lightning as pl

from typing import Dict
from torch.utils.data import DataLoader

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from einops import rearrange


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def visualize_and_save_heatmap(heat_map, key_word=''):
    heat_map = heat_map.cpu().numpy()
    #normalize the heat_map to 0-255
    heat_map = (heat_map - heat_map.min()) / (heat_map.max() - heat_map.min())
    heat_map = (heat_map * 255).astype(np.uint8)
    heat_map = cv2.resize(heat_map, (512, 512))

    heatmap_color = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
    print(key_word)
    cv2.imwrite(f"{key_word}_heatmap.png", heatmap_color)



def visualize_and_save_features_pca(feats_map, key_word=''):
    """
    feats_map: [B, N, D]
    """
    B = len(feats_map)

    if len(feats_map.shape)==3:
        h = w = int(np.sqrt(feats_map.shape[1]))
        arr = rearrange(feats_map, 'b (h w) c -> b c h w', h=h, w=w)
        arr = arr.squeeze(0).cpu().numpy()

    else:
        h = w = int(feats_map.shape[2])
        arr = feats_map.squeeze(0).cpu().numpy()
        feats_map = rearrange(feats_map, 'b c h w -> b (h w) c', h=h, w=w)

    #np.save(f'{key_word}.npy', arr)


    feats_map = feats_map.flatten(0, -2)
    feats_map = feats_map.cpu().numpy()
    
    pca = PCA(n_components=3)
    pca.fit(feats_map)
    feature_maps_pca = pca.transform(feats_map)  # N X 3
    feature_maps_pca = feature_maps_pca.reshape(B, -1, 3)  # B x (H * W) x 3
    for i, experiment in enumerate(feature_maps_pca):
        pca_img = feature_maps_pca[i]  # (H * W) x 3
        h = w = int(np.sqrt(pca_img.shape[0]))
        pca_img = pca_img.reshape(h, w, 3)
        pca_img_min = pca_img.min(axis=(0, 1))
        pca_img_max = pca_img.max(axis=(0, 1))
        pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)
        pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
        pca_img = pca_img.resize((512, 512))
        pca_img.save(f"{key_word}_pca.png")

def visualize_and_save_features_kmean(feats_map, key_word=''):

    B = len(feats_map)
    feats_map = feats_map.flatten(0, -2)
    feats_map = feats_map.cpu().numpy()

    # K-means 聚类
    n_clusters = 4  # 簇数
    '''kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    kmeans.fit(feats_map)
    # 聚类标签
    labels = kmeans.labels_'''



    # 使用 'ward' 方法进行层次聚类（Ward 法是常用的）
    Z = linkage(feats_map, method='ward')

    # 使用 fcluster 提取簇，t 是阈值，criterion 指定聚类依据
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')


    h = w = int(np.sqrt(labels.shape[0]))

    # 重塑回空间维度
    clustered_map = labels.reshape(h, w)

    cmap = plt.get_cmap('tab20', n_clusters)  # 'tab20'是一种离散调色板，适合分类
    
    # 为每个类别生成RGB颜色，并将颜色值从0-1转换为0-255
    colors = (cmap(np.arange(n_clusters))[:, :3] * 255).astype(np.uint8)

    # 创建彩色分割图
    color_seg = np.zeros((clustered_map.shape[0], clustered_map.shape[1], 3), dtype=np.uint8)
    for class_id in range(n_clusters):
        mask = (clustered_map == class_id)
        color_seg[mask] = colors[class_id]

    color_seg = Image.fromarray(color_seg.astype(np.uint8))
    color_seg = color_seg.resize((512, 512))
    color_seg.save(f"{key_word}_hcluster.png")

'''def visualize_and_save_features_kmean(feats_map, save_dir='visiualization/segmentor/layer/', key_word=''):

    B = len(feats_map)
    feats_map = feats_map.flatten(0, -2)
    feats_map = feats_map.cpu().numpy()

    # K-means 聚类
    n_clusters = 5  # 簇数
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    kmeans.fit(feats_map)

    # 聚类标签
    labels = kmeans.labels_

    h = w = int(np.sqrt(labels.shape[0]))

    # 重塑回空间维度
    clustered_map = labels.reshape(h, w)

    clustered_map += 1
    clustered_map *= 50
    clustered_map += 50
    clustered_map = Image.fromarray(clustered_map.astype(np.uint8))
    clustered_map = clustered_map.resize((512, 512))
    clustered_map.save(os.path.join(save_dir, f"{key_word}_kmean.png"))'''


def DataloaderFromConfig(batch_size, train=None, validation=None,
                 num_workers=None, shuffle_val_dataloader=False):
    if train is not None:
        train_dataset = instantiate_from_config(train)
        train_dataloader = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    if validation is not None:
        val_dataset = instantiate_from_config(validation)
        val_dataloader = DataLoader(val_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle_val_dataloader)
        
    return train_dataloader, val_dataloader

def read_label(csv_label_path, label_dic={}):
    csvFile = open(csv_label_path, "r")
    reader = csv.reader(csvFile)
    for item in reader:
        if reader.line_num == 1:
            continue
        label_dic[item[0]] = [item[-2], item[-1]]
    return label_dic

def makedir(path):
    # Split the path into directories at each level according to '/'
    path_parts = path.split('/')
    current_path = ''
    # Determine whether the path exists step by step, and create it if it does not exist
    for part in path_parts:
        # Splice the current path with the current level directory
        current_path = os.path.join(current_path, part)
        # If the current path does not exist, create it
        if not os.path.exists(current_path):
            os.makedirs(current_path)

def is_overlaped(array_1, array_2):
    x_min, y_min, x_max, y_max = array_2[0]
    x_min -= 16
    y_min -= 16
    x_max += 16
    y_max += 16
    
    x_min_n = array_1[:, 0]
    y_min_n = array_1[:, 1]
    x_max_n = array_1[:, 2]
    y_max_n = array_1[:, 3]
    intersect = ((x_min <= x_max_n) & (x_max >= x_min_n) & (y_min <= y_max_n) & (y_max >= y_min_n))
    
    return intersect

def get_layout_image(img, laBel):
    h, w, c = img.shape 
    bboxes = []
    isolated_boxes = []
    overlaped_boxes = []

    box_lay_1 = []
    box_lay_2 = []
    box_lay_3 = []
    box_lay_extra = []

    if "no_box" in laBel:
        source_img = np.zeros((h, w, 3), dtype=np.uint8)
        n_lay = 0
        return source_img, n_lay

    else:
        for bbox in laBel:
            bbox = np.array(list(map(int,bbox.split(','))))
            bboxes.append(bbox)
        bboxes = np.array(bboxes)
        for bbox in bboxes:
            bbox = bbox.reshape(1, 4)
            overlap = is_overlaped(bboxes, bbox)
            n_box_overlap = np.sum(overlap.astype(int))
            bbox = bbox.reshape(4)
            if np.sum(n_box_overlap)==1:
                isolated_boxes.append(bbox)
            else:
                overlaped_boxes.append(bbox)

        lays = [0]   #放置box的层数
        if len(overlaped_boxes) > 0:
            overlaped_boxes = np.array(overlaped_boxes)
            overlaped_matrix = None
            for bbox in overlaped_boxes:
                bbox = bbox.reshape(1, 4)
                overlap = is_overlaped(overlaped_boxes, bbox)
                overlap = overlap.reshape(1, overlap.size)
                if overlaped_matrix is not None:
                    overlaped_matrix = np.concatenate((overlaped_matrix, overlap), axis=0) 
                else:
                    overlaped_matrix = overlap
            
            diagonal_matrix = abs(np.eye(overlap.size) - 1)
            diagonal_matrix = diagonal_matrix.astype(bool)
            overlaped_matrix *= diagonal_matrix
            overlaped_matrix = overlaped_matrix.astype(bool)

            G = nx.Graph(overlaped_matrix)
            coloring = nx.coloring.greedy_color(G, strategy='largest_first')

            for box_id, lay_id in coloring.items():
                lays.append(lay_id)   
                if lay_id == 0:
                    box_lay_1.append(overlaped_boxes[box_id])
                if lay_id == 1:
                    box_lay_2.append(overlaped_boxes[box_id])
                if lay_id == 2:
                    box_lay_3.append(overlaped_boxes[box_id])
                if lay_id > 2:
                    box_lay_extra.append(overlaped_boxes[box_id])

        n_lay = np.max(np.array(lays)) + 1

        box_lay_1 = np.array(box_lay_1)
        box_lay_2 = np.array(box_lay_2)
        box_lay_3 = np.array(box_lay_3)

        
        isolated_boxes = np.array(isolated_boxes)
        box_lay_extra = np.array(box_lay_extra)


        if isolated_boxes.shape[0]:
            if box_lay_1.size != 0:
                box_lay_1 = np.concatenate((box_lay_1, isolated_boxes), axis=0)
            else:
                box_lay_1 = isolated_boxes

        if box_lay_extra.shape[0]:
            if box_lay_extra.shape[0] > 1:
                indices = np.random.choice(range(len(box_lay_extra)), size=len(box_lay_extra)//2, replace=False)
                arr1 = box_lay_extra[indices]
                arr2 = np.delete(box_lay_extra, indices, axis=0)
                box_lay_2 = np.concatenate((box_lay_2, arr1), axis=0)
                box_lay_3 = np.concatenate((box_lay_3, arr2), axis=0)
            else:
                box_lay_3 = np.concatenate((box_lay_3, box_lay_extra), axis=0)


        #draw bbox
        box_img_lay_1 = np.zeros((h, w, 1), dtype=np.uint8)
        box_img_lay_2 = np.zeros((h, w, 1), dtype=np.uint8)
        box_img_lay_3 = np.zeros((h, w, 1), dtype=np.uint8)

        if box_lay_1.shape[0] > 1:
            for bbox in box_lay_1:
                cv2.rectangle(box_img_lay_1, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255), -1)
        if box_lay_2.shape[0] > 1:
            for bbox in box_lay_2:
                cv2.rectangle(box_img_lay_2, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255), -1)
        if box_lay_3.shape[0] > 1:
            for bbox in box_lay_3:
                if bbox not in box_lay_2:
                    cv2.rectangle(box_img_lay_3, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255), -1)        
                
        source_img = np.concatenate((box_img_lay_1, box_img_lay_2, box_img_lay_3), axis=2)

    return source_img, n_lay
        
class SaveLoraOnlyCallback(pl.Callback):
    def __init__(self, save_lora_only, paras_to_save='lora_', save_path='lora_model', filename='lora'):
        self.save_lora_only = save_lora_only
        self.paras_to_save = paras_to_save
        self.save_path = save_path
        self.filename = filename

    def on_epoch_end(self, trainer, pl_module):
        # 创建一个新的字典，只包含要保存的参数
        if self.save_lora_only:
            state_dict_to_save = {key: value for key, value in pl_module.state_dict().items() if self.paras_to_save in key}
        else:
            state_dict_to_save = pl_module.state_dict()
        # 拼接保存的文件路径
        save_filepath = f"{self.save_path}/{self.filename}_epoch{trainer.current_epoch}.pth"

        # 保存权重
        torch.save(state_dict_to_save, save_filepath)

def rand_rotate(img, mask):
    #创建正方形的四个顶点坐标和中心坐标
    h, w, c = img.shape
    square_size = h
    center = (h//2, h//2)
    square_points = np.array([(0, 0), (0, square_size), (square_size, 0), (square_size, square_size)], dtype=np.float32)

    #随机生成旋转角度
    angle = random.uniform(-90, 90)

    #创建顶点旋转矩阵并应用旋转
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_points = cv2.transform(square_points.reshape(1, -1, 2), rotation_matrix)

    y_min = int(np.min(rotated_points[0, :, 1]))
    y_max = int(np.max(rotated_points[0, :, 1]))
    max_scale = square_size/(y_max - y_min)

    #将缩放因子设置为（0.5，max_scale)
    scale_factor = random.uniform(0.6, max_scale)

    # 创建缩放矩阵
    scaling_matrix = np.array([[scale_factor, 0, (1 - scale_factor) * center[0]],
                            [0, scale_factor, (1 - scale_factor) * center[1]]], dtype=np.float32)

    # 应用缩放
    scaled_points = cv2.transform(rotated_points, scaling_matrix)

    # 将坐标四舍五入为整数
    scaled_points = np.int32(scaled_points)

    transformed_x_min = int(np.min(scaled_points[:, :, 0]))
    transformed_y_min = int(np.min(scaled_points[:, :, 1]))

    #随机生成平移量，并平移
    shift_x = random.randint(-transformed_x_min, transformed_x_min)
    shift_y = random.randint(-transformed_y_min, transformed_y_min)

    scaled_points[:, :, 0] += shift_x
    scaled_points[:, :, 1] += shift_y

    back_rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

    # 执行旋转
    rotated_img = cv2.warpAffine(img, back_rotation_matrix, (h, w))
    rotated_mask = cv2.warpAffine(mask, back_rotation_matrix, (h, w))

    scaled_points = cv2.transform(scaled_points.reshape(1, -1, 2), back_rotation_matrix)
    scaled_points = np.int32(scaled_points)

    x_max = np.max(scaled_points[:, :, 0])
    x_min = np.min(scaled_points[:, :, 0])
    y_max = np.max(scaled_points[:, :, 1])
    y_min = np.min(scaled_points[:, :, 1])

    return rotated_img[y_min:y_max, x_min:x_max], rotated_mask[y_min:y_max, x_min:x_max]

def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model

def create_dataloader(config_path):
    config = OmegaConf.load(config_path)
    data = instantiate_from_config(config.data)
    return data
