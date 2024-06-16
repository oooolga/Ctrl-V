from ctrlv.utils import get_dataloader, plot_3d_bbox, save_image
import torch, os
from tqdm import tqdm

import argparse
import cv2
import numpy as np
import os
from functools import partial
from pathlib import Path
import sys

# # replace the root to the directory where your dataset folders are located
root = sys.argv[1] # <- set root to the directory where your dataset folders are located

# download bdd100k dataset from https://www.vis.xyz/bdd100k/ and extract it to the root directory
print('Start preprocessing bdd100k dataset.')
for mode in ['train', 'val']:
    dataset, _ = get_dataloader(root, 'bdd100k', if_train=bool(mode=='train'), batch_size=1, num_workers=1, 
                                data_type='image', use_default_collate=True, tokenizer=None, shuffle=False)
    bbox_dir = dataset.image_dir.replace(dataset.TO_IMAGE_DIR, dataset.TO_BBOX_DIR)
    for clip_dir in dataset.clip_folders:
        os.makedirs(os.path.join(bbox_dir, clip_dir), exist_ok=True)

    for i in tqdm(range(len(dataset))):
        pixel_value, objects, _, _ = dataset.__getitem__(i, return_prompt=True, return_calib=False, return_index=True)
        pixel_value = dataset.revert_transform(pixel_value)
        canvas = torch.zeros_like(pixel_value)
        bbox_im = plot_3d_bbox(canvas, objects, None, plot_2d_bbox=True)
        save_path = dataset.get_image_file_by_index(i).replace(dataset.TO_IMAGE_DIR, dataset.TO_BBOX_DIR)
        save_image(bbox_im, save_path, verbose=False)

print('Start preprocessing kitti dataset.')
dataset, _ = get_dataloader(root, 'kitti', if_train=True, batch_size=1, num_workers=1, 
                            data_type='image', use_default_collate=True, tokenizer=None, shuffle=False)

bbox_dir = os.path.join(dataset.root, dataset.version, dataset._location, dataset.TO_BBOX_DIR)

image_dir = os.path.join(dataset.root, dataset.version, dataset._location, dataset.TO_IMAGE_DIR)
for scene in sorted(os.listdir(image_dir)):
    scene_dir = os.path.join(bbox_dir, scene)
    os.makedirs(scene_dir, exist_ok=True)

for i in tqdm(range(len(dataset))):
    
    pixel_value, objects, _, cam_to_img, _ = dataset.__getitem__(i, return_prompt=True, return_calib=True, return_index=True)
    pixel_value = dataset.revert_transform(pixel_value)
    canvas = torch.zeros_like(pixel_value)
    bbox_im = plot_3d_bbox(canvas, objects, cam_to_img, plot_2d_bbox=True)
    save_path = dataset.get_image_file_by_index(i).replace(dataset.TO_IMAGE_DIR, dataset.TO_BBOX_DIR)
    save_image(bbox_im, save_path, verbose=False)

print('Finished preprocessing kitti dataset.')

print('Start preprocessing vkitti dataset.')
for mode in ['train', 'val']:
    dataset, _ = get_dataloader(root, 'vkitti', if_train=bool(mode=='train'), batch_size=1, num_workers=1, 
                                data_type='image', use_default_collate=True, tokenizer=None, shuffle=False)
    folder_dirs = dataset.VERSIONS_DIRS[dataset.version]
    img_dir_prefix = os.path.join(dataset.root, dataset.version, folder_dirs['TOP'][0])

    for subfolder in folder_dirs['TRAINING'] if mode == 'train' else folder_dirs['TESTING']:
        scene = {}
        for setting in folder_dirs['SETTINGS']:
            setting_images = []
            image_dir = os.path.join(img_dir_prefix, subfolder, setting, folder_dirs['TO_RGB'])
            bbox_dir = image_dir.replace('rgb', 'bbox')
            os.makedirs(bbox_dir, exist_ok=True)
    for i in tqdm(range(len(dataset))):
        pixel_value, objects, _, cam_to_img, _ = dataset.__getitem__(i, return_prompt=True, return_calib=True, return_index=True)
        pixel_value = dataset.revert_transform(pixel_value)
        canvas = torch.zeros_like(pixel_value)
        bbox_im = plot_3d_bbox(canvas, objects, cam_to_img, plot_2d_bbox=True)
        save_path = dataset.get_image_file_by_index(i).replace('rgb', 'bbox')
        save_image(bbox_im, save_path, verbose=False)
print('Finished preprocessing vkitti dataset.')