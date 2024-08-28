from PIL import Image
import torch
import pandas as pd
from .kitti_abstract import KittiAbstract

import os

class VKittiDataset(KittiAbstract):
    VERSIONS = ['vkitti_1.3.1', 'vkitti_2.0.3']
    VERSIONS_DIRS = {
        'vkitti_1.3.1': {
            'TOP': ['vkitti_1.3.1_rgb', 'vkitti_1.3.1_motgt'],
            'SUBS': ['0001', '0002', '0006', '0018', '0020'],
            'SETTINGS': ['clone', 'fog', 'morning', 'overcast', 'rain', 'sunset'],
            'TO_RGB': '.',
            'TRAINING': ['0001', '0002', '0006', '0018'],
            'TESTING': ['0020'],
        },
        'vkitti_2.0.3': {
            'TOP': ['rgb', 'textgt', 'bbox'],
            'SUBS': ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20'],
            'SETTINGS': ['clone', 'fog', 'morning', 'overcast', 'rain', 'sunset'],
            'TO_RGB': 'frames/rgb/Camera_0',
            'TO_BBOX': 'frames/bbox/Camera_0',
            'TRAINING': ['Scene01', 'Scene02', 'Scene06', 'Scene18'],
            'TESTING': ['Scene20'],
            'BBOX_FILENAME': 'bbox.txt',
            'OBJECT_FILENAME': 'info.txt',
            'POSE_FILENAME': 'pose.txt',
            'INTRINSIC_FILENAME': 'intrinsic.txt',
            'EXTRINSIC_FILENAME': 'extrinsic.txt',
        }
    }
    SCENE_LOOKUP = {'Scene01': 'Crowded urban area',
                    'Scene02': 'Urban area',
                    'Scene06': 'Busy intersection',
                    'Scene18': 'Long road in the forest',
                    'Scene20': 'Highway'}
    PIXEL_THRES = 350

    TO_COCO_LABELS = {
        1: 2,
        2: 2,
        3: 7,
        7: 6,
    }

    def __init__(self,
                 root="./datasets",
                 version='vkitti_2.0.3',
                 train=True,
                 target_transform=None,
                 data_type="image",
                 clip_length=None,
                 if_return_prompt=True,
                 if_return_index=True,
                 if_return_calib=False,
                 if_return_bbox_im=False,
                 train_H=None, train_W=None,
                 use_preplotted_bbox=True,
                 non_overlapping_clips=False):
        
        super(VKittiDataset, self).__init__(root=root, 
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
        
        self.version = version
        if self.version != "vkitti_2.0.3":
            raise NotImplementedError("Only vkitti_2.0.3 is supported")
        
        if self.data_type == 'clip':
            self.clip_list = []
        folder_dirs = VKittiDataset.VERSIONS_DIRS[self.version]
        img_dir_prefix = os.path.join(self.root, self.version, folder_dirs['TOP'][0])
        for subfolder in folder_dirs['TRAINING'] if self.train else folder_dirs['TESTING']:
            scene = {}
            for setting in folder_dirs['SETTINGS']:
                setting_images = []
                image_dir = os.path.join(img_dir_prefix, subfolder, setting, folder_dirs['TO_RGB'])
                for image_file in sorted(os.listdir(image_dir)):
                    self.image_list.append(os.path.join(image_dir, image_file))
                    setting_images.append(len(self.image_list)-1)
                scene[setting] = setting_images
                if self.data_type == 'clip':
                    if not non_overlapping_clips:
                        for image_idx in range(len(scene[setting])-self.clip_length):
                            self.clip_list.append(scene[setting][image_idx:image_idx+self.clip_length])
                    else:
                        # In case self.clip_length << actual video sample length, we can create multiple non-overlapping clips for each sample    
                        total_frames = len(scene[setting])
                        for clip_i in range(total_frames // self.clip_length):
                            start_image_idx = clip_i * self.clip_length
                            end_image_idx = start_image_idx + self.clip_length
                            self.clip_list.append(scene[setting][start_image_idx:end_image_idx])
            self.images[subfolder] = scene


    def __len__(self):
        return len(self.image_list) if self.data_type == 'image' else len(self.clip_list)
        
    def _getimageitem(self, index, return_prompt=False, return_calib=False, return_index=False, return_bbox_im=False):
        # get the image        
        image_file = self.get_image_file_by_index(index)
        image = Image.open(image_file)
        # get the target
        label_folder = image_file.replace(VKittiDataset.VERSIONS_DIRS[self.version]['TOP'][0],
                                                      VKittiDataset.VERSIONS_DIRS[self.version]['TOP'][1], 1)
        label_folder = label_folder[:label_folder.find(VKittiDataset.VERSIONS_DIRS[self.version]['TO_RGB'])]
        frame_id = int(image_file.split('/')[-1].split('.')[0].split('_')[1])
        try:
            target = self._parse_label(label_folder, frame_id)
        except:
            target = []
            
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        ret = (image, target, )
        
        if return_prompt or self.if_return_prompt:
            prompt = self.prompt_engineer(index)
            ret += (prompt, )
        if return_calib or self.if_return_calib:
            # get the camera calibration information
            calib = self._parse_calib(label_folder, frame_id)
            ret += (calib, )
        if return_index or self.if_return_index:
            ret += (index,)
        if return_bbox_im or self.if_return_bbox_im:
            bbox_file = self.get_bbox_image_file_by_index(image_file=image_file)
            bbox_im = Image.open(bbox_file)
            if not self.transform is None:
                bbox_im = self.transform(bbox_im)
            ret += (bbox_im, )
        return ret
    
    def _parse_label(self, label_folder, frame_id):
        bbox_file = os.path.join(label_folder, VKittiDataset.VERSIONS_DIRS[self.version]['BBOX_FILENAME'])
        object_file = os.path.join(label_folder, VKittiDataset.VERSIONS_DIRS[self.version]['OBJECT_FILENAME'])
        pose_file = os.path.join(label_folder, VKittiDataset.VERSIONS_DIRS[self.version]['POSE_FILENAME'])
        target = []

        bbox_df = pd.read_csv(bbox_file, sep=' ')
        object_df = pd.read_csv(object_file, sep=' ')
        pose_df = pd.read_csv(pose_file, sep=' ')
        # select frame 
        bbox_df = bbox_df.loc[bbox_df['frame'] == frame_id]
        bbox_df = bbox_df.loc[bbox_df['cameraID'] == 0]
        pose_df = pose_df.loc[pose_df['frame'] == frame_id]
        pose_df = pose_df.loc[pose_df['cameraID'] == 0]
        bbox_count = 0
        target = []
        for _, row in bbox_df.iterrows():
            object_type = object_df.loc[object_df['trackID'] == row['trackID']]['label'].values[0]
            if object_type == 'DontCare': continue;
            if row['number_pixels'] <= VKittiDataset.PIXEL_THRES: continue;
            pose = pose_df.loc[pose_df['trackID']==row['trackID']]
            assert len(pose) == 1
            pose = pose.squeeze(axis=0)
            target.append(
                {   
                    'frame': frame_id,
                    'trackID': row['trackID'],
                    'type': object_type,
                    'truncated': row['truncation_ratio'],
                    'occluded': row['occupancy_ratio'],
                    'alpha': pose['alpha'],
                    'bbox': [row['left'], row['top'], row['right'], row['bottom']],
                    'dimensions': [pose['height'], pose['width'], pose['length']],
                    'location': [pose['camera_space_X'], pose['camera_space_Y'], pose['camera_space_Z']],
                    'rotation_y': pose['rotation_camera_space_y'],
                    'id_type': VKittiDataset.CLASS_IDS_LOOKUP[object_type],
                }
            )
            bbox_count += 1
            if bbox_count >= self.MAX_BOXES_PER_DATA:
                break
        return target
    
    def _parse_calib(self, label_folder, frame_id):
        intrinsic_file = os.path.join(label_folder, VKittiDataset.VERSIONS_DIRS[self.version]['INTRINSIC_FILENAME'])
        intrinsic_df = pd.read_csv(intrinsic_file, sep=' ')
        intrinsic_df = intrinsic_df.loc[intrinsic_df['frame'] == frame_id]
        intrinsic_df = intrinsic_df.loc[intrinsic_df['cameraID'] == 0]
        import numpy as np
        RET = np.zeros((3,3))
        RET[0,0] = intrinsic_df['K[0,0]'].values[0]
        RET[0,2] = intrinsic_df['K[0,2]'].values[0]
        RET[1,1] = intrinsic_df['K[1,1]'].values[0]
        RET[1,2] = intrinsic_df['K[1,2]'].values[0]
        RET[2,2] = 1
        return RET

    def get_bbox_image_file_by_index(self, index=None, image_file=None):
        if image_file is None:
            image_file = self.get_image_file_by_index(index)
        return image_file.replace('rgb', 'bbox')
    
    def prompt_engineer(self, index):
        for setting in VKittiDataset.VERSIONS_DIRS[self.version]['SETTINGS']:
            if self.image_list[index].find(setting) != -1:
                break
        scene_idx = self.image_list[index].find('Scene')
        scene = VKittiDataset.SCENE_LOOKUP[self.image_list[index][scene_idx:scene_idx+7]]
        return "This is a simulated driving scene set in a {} {} {}.".format(scene.lower(),
                                                                            'in the' if setting in ['morning', 'rain', 'fog'] else 'during',
                                                                            setting if setting != 'clone' else 'daytime')

if __name__ == "__main__":
    # vkitti1 = VKittiDataset(train=True,
    #                         version='vkitti_1.3.1')
    
    vkitti2 = VKittiDataset(train=True)
    print(vkitti2.__getitem__(0, return_prompt=True))

    from ctrlv.datasets import kitti_collate_fn
    collate_fn = kitti_collate_fn

    dataloader = torch.utils.data.DataLoader(vkitti2, batch_size=5, shuffle=True, collate_fn=collate_fn)
    import pdb; pdb.set_trace()