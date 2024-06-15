from PIL import Image
import torch
import pandas as pd
from .kitti_abstract import KittiAbstract

import os

class KittiDataset(KittiAbstract):

    # Values   Name         Description
    # ----------------------------------------------------------------------------
    #     1    type         Describes the type of object: 'Car', 'Van', 'Truck',
    #                         'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
    #                         'Misc' or 'DontCare'
    #     1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
    #                         truncated refers to the object leaving image boundaries
    #     1    occluded     Integer (0,1,2,3) indicating occlusion state:
    #                         0 = fully visible, 1 = partly occluded
    #                         2 = largely occluded, 3 = unknown
    #     1    alpha        Observation angle of object, ranging [-pi..pi]
    #     4    bbox         2D bounding box of object in the image (0-based index):
    #                         contains left, top, right, bottom pixel coordinates
    #     3    dimensions   3D object dimensions: height, width, length (in meters)
    #     3    location     3D object location x,y,z in camera coordinates (in meters)
    #     1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
    #     1    score        Only for results: Float, indicating confidence in
    #                         detection, needed for p/r curves, higher is better.
    
    TO_IMAGE_DIR = "image_02"
    TO_LABEL_DIR = "label_02"
    TO_BBOX_DIR = "bbox_02"
    TO_CALIB_DIR = "calib"
    TRAIN_SPLIT = [f'{i:04d}' for i in range(19)]
    TEST_SPLIT = [f'{i:04d}' for i in [19, 20]]

    def __init__(self,
                 root='./datasets',
                 version='kitti',
                 train=True,
                 target_transform=None,
                 data_type='image',
                 clip_length=None,
                 if_return_prompt=True,
                 if_return_index=True,
                 if_return_calib=False,
                 if_return_bbox_im=False,
                 train_H=None, train_W=None,
                 use_preplotted_bbox=True):

        super(KittiDataset, self).__init__(root=root, 
                                           train=train, 
                                           target_transform=target_transform, 
                                           data_type=data_type, 
                                           clip_length=clip_length,
                                           if_return_prompt=if_return_prompt,
                                           if_return_index=if_return_index,
                                           if_return_calib=if_return_calib,
                                           if_return_bbox_im=if_return_bbox_im,
                                           train_H=train_H, train_W=train_W,
                                           use_preplotted_bbox=use_preplotted_bbox)
        self._location = 'training'# if self.train else 'testing'
        self.version = version
        
        if self.data_type == 'clip':
            self.clip_list = []
        image_dir = os.path.join(self.root, self.version, self._location, KittiDataset.TO_IMAGE_DIR)
        _split = KittiDataset.TRAIN_SPLIT if self.train else KittiDataset.TEST_SPLIT
        for scene in _split:
            scene_dir = os.path.join(image_dir, scene)
            self.images[scene] = []
            for image_file in sorted(os.listdir(scene_dir)):
                self.image_list.append(os.path.join(scene_dir, image_file))
                self.images[scene].append(len(self.image_list)-1)
            if self.data_type == 'clip':
                for image_idx in range(len(self.images[scene])-self.clip_length):
                    self.clip_list.append(self.images[scene][image_idx:image_idx+self.clip_length])
        

    def _getimageitem(self, index, return_prompt=False, return_calib=False, return_index=False, return_bbox_im=False):
        # get the image
        image_file = self.get_image_file_by_index(index)
        image = Image.open(image_file) if self.if_return_im else None
        # get the target
        label_file = image_file.replace(KittiDataset.TO_IMAGE_DIR, KittiDataset.TO_LABEL_DIR)
        frame_id = int(label_file.split('/')[-1].split('.')[0])
        label_file = label_file[:label_file.rfind('/')]+'.txt'
        target = self._parse_label(label_file, frame_id) # if self.train else None

        if self.if_return_im and not self.transforms is None:
            image, target = self.transforms(image, target)

        ret = (image, target, )
        if return_prompt or self.if_return_prompt:
            prompt = self.prompt_engineer(index)
            ret += (prompt, )
        
        if return_calib or self.if_return_calib:
            # get the camera calibration information
            calib_file = label_file.replace('label_02', 'calib')
            calib = self._parse_calib(calib_file)
            ret += (calib, )

        if return_index or self.if_return_index:
            ret += (index, )
        
        if return_bbox_im or self.if_return_bbox_im:
            bbox_file = self.get_bbox_image_file_by_index(image_file=image_file)
            bbox_im = Image.open(bbox_file)
            if not self.transform is None:
                bbox_im = self.transform(bbox_im)
            ret += (bbox_im, )
        return ret
    
    def _parse_label(self, label_file, frame_id):
        target = []

        bbox_df = pd.read_csv(label_file, sep=' ', header=None,
                              names=['frame', 'trackID', 'type', 'truncated', 'occluded', 'alpha', 
                                     'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom', 
                                     'height', 'width', 'length', 'X', 'Y', 'Z', 'rotation_y',])
        bbox_df = bbox_df.loc[bbox_df['frame'] == frame_id]
        bbox_df = bbox_df.loc[bbox_df['type'] != 'DontCare']
        bbox_count = 0
        for _, row in bbox_df.iterrows():
            target.append(
                {
                    'frame': frame_id,
                    'trackID': row['trackID'],
                    'type': row['type'],
                    'truncated': row['truncated'],
                    'occluded': row['occluded'],
                    'alpha': row['alpha'],
                    'bbox': [row['bbox_left'], row['bbox_top'], row['bbox_right'], row['bbox_bottom']],
                    'dimensions': [row['height'], row['width'], row['length']],
                    'location': [row['X'], row['Y'], row['Z']],
                    'rotation_y': row['rotation_y'],
                    'id_type': KittiDataset.CLASS_IDS_LOOKUP[row['type']],
                }
            )
            bbox_count += 1
            if bbox_count >= self.MAX_BOXES_PER_DATA:
                break
        return target
    
    def _parse_calib(self, calib_file):
        import numpy as np
        for line in open(calib_file, 'r'):
            if 'P2:' in line:
                cam_to_img = line.strip().split(' ')
                cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
                cam_to_img = np.reshape(cam_to_img, (3, 4))
                return cam_to_img

    def get_bbox_image_file_by_index(self, index=None, image_file=None):
        if image_file is None:
            image_file = self.get_image_file_by_index(index)
        return image_file.replace(KittiDataset.TO_IMAGE_DIR, KittiDataset.TO_BBOX_DIR)
    
    def prompt_engineer(self, *args):
        return "This is a real-world driving scene set in the German city of Karlsruhe."
    
    def __len__(self):
        return len(self.image_list) if self.data_type == 'image' else len(self.clip_list)

if __name__ == '__main__':
    
    dataset = KittiDataset(train=True)
    print(len(dataset))
    print(dataset.__getitem__(0))

    from sd3d.datasets import kitti_collate_fn
    collate_fn = kitti_collate_fn

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=collate_fn)

    dataset = KittiDataset(train=True, data_type='clip', clip_length=10)