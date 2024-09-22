from .kitti_abstract import KittiAbstract

from PIL import Image
import torch
import os
import json
import random

class BDD100KDataset(KittiAbstract):
    IDS_CLASS_LOOKUP = {
        1: 'pedestrian',
        2: 'rider',
        3: 'car',
        4: 'truck',
        5: 'bus',
        6: 'train',
        7: 'motorcycle',
        8: 'bicycle',
        9: 'traffic light',
        10: 'traffic sign',
    }

    CLASS_IDS_LOOKUP = {
        'pedestrian': 1,
        'rider': 2,
        'car': 3,
        'truck': 4,
        'bus': 5,
        'train': 6,
        'motorcycle': 7,
        'bicycle': 8,
        'traffic light': 9,
        'traffic sign': 10,
    }

    TO_COCO_LABELS = {
        1: 0,
        2: 0,
        3: 2,
        4: 7,
        5: 5,
        6: 6,
    }

    TO_IMAGE_DIR = 'images/track'
    TO_BBOX_DIR = 'bboxes/track'
    TO_LABEL_DIR = 'labels'
    TO_BBOX_LABELS = 'labels/box_track_20'
    TO_SEG_LABELS = 'labels/seg_track_20/colormaps'
    TO_POSE_LABELS = 'labels/pose_21' # not very useful, not every image has a pose label

    def __init__(self,
                 root='./datasets', 
                 train=True,
                 target_transform=None,
                 data_type='image',
                 clip_length=None,
                 if_return_prompt=True,
                 if_return_index=True,
                 if_return_calib=False,
                 if_return_bbox_im=False,
                 H=None, W=None,
                 train_H=None, train_W=None,
                 use_segmentation=False,
                 use_preplotted_bbox=True):

        super(BDD100KDataset, self).__init__(root=root, 
                                     train=train, 
                                     target_transform=target_transform, 
                                     data_type=data_type, 
                                     clip_length=clip_length,
                                     if_return_prompt=if_return_prompt,
                                     if_return_index=if_return_index,
                                     if_return_calib=if_return_calib,
                                     if_return_bbox_im=if_return_bbox_im,
                                     H=720 if H is None else H,
                                     W=1280 if W is None else W,
                                     train_H=train_H, train_W=train_W,
                                     use_preplotted_bbox=use_preplotted_bbox)
        
        self.MAX_BOXES_PER_DATA = 30
        self._location = 'train' if self.train else 'val'
        self.version = 'bdd100k'
        self.use_segmentation = use_segmentation

        self.image_dir = os.path.join(self.root, self.version, BDD100KDataset.TO_IMAGE_DIR, self._location)
        self.bbox_label_dir = os.path.join(self.root, self.version, BDD100KDataset.TO_BBOX_LABELS, self._location)

        if not self.use_segmentation:
            listed_image_dir = os.listdir(self.image_dir)
            try:
                listed_image_dir.remove('pred')
            except:
                pass
            self.clip_folders = sorted(listed_image_dir)
            self.clip_folder_lengths = {k:len(os.listdir(os.path.join(self.image_dir, k))) for k in self.clip_folders}
        else:
            seg_label_dir = os.path.join(self.root, self.version, BDD100KDataset.TO_SEG_LABELS, self._location)
            self.clip_folders = sorted(os.listdir(seg_label_dir))
            self.clip_folder_lengths = {k:len(os.listdir(os.path.join(seg_label_dir, k))) for k in self.clip_folders}

        if self.data_type == 'clip':
            for l in self.clip_folder_lengths.values():
                assert l >= self.clip_length, f'clip length {self.clip_length} is too long for clip folder length {l}'
    
    def _getimageitem(self, index, return_prompt=False, return_calib=False, return_index=False, return_bbox_im=False):
        # get the image
        image_file = self.get_image_file_by_index(index)
        image = Image.open(image_file)
        clip_id = image_file[:image_file.rfind('/')]
        clip_id = clip_id[clip_id.rfind('/')+1:]
        label_file = os.path.join(self.bbox_label_dir, f'{clip_id}.json')
        
        frame_id = image_file[image_file.rfind('/')+1:]
        target = self._parse_label(label_file, frame_id)

        if not self.transforms is None:
            image, target = self.transforms(image, target)

        ret = (image, target, )
        if return_prompt or self.if_return_prompt:
            prompt = self.prompt_engineer()
            ret += (prompt, )
        
        if return_calib or self.if_return_calib:
            # get the camera calibration information
            ret += (None, )
        
        if return_index or self.if_return_index:
            ret += (index, )
        if return_bbox_im or self.if_return_bbox_im:
            if self.use_preplotted_bbox or self.use_segmentation:
                bbox_file = self.get_bbox_image_file_by_index(image_file=image_file)
                bbox_im = Image.open(bbox_file)
                if self.use_segmentation:
                    bbox_im = bbox_im.convert('RGB')
                if not self.transform is None:
                    bbox_im = self.transform(bbox_im)
            else:
                bbox_im = self._draw_bbox(target, None)
            ret += (bbox_im, )
        return ret
    
    def _getclipitem(self, index, return_prompt=False, return_calib=False, return_index=False, return_bbox_im=False):
        frames = self.get_clip_frame_indices(index)
        images = []
        targets = []
        bboxes = []
        
        if_return_bbox_im_cp = self.if_return_bbox_im
        self.set_if_return_bbox_im(False)
        if not self.if_return_bbox_im:
            prompt = self.prompt_engineer()
        
        self.disable_all_settings()
        traj = None
        for frame_i, frame in enumerate(frames):
            if not (if_return_bbox_im_cp or return_bbox_im):
                image, target = self._getimageitem(frame, return_prompt=False, return_calib=False, return_index=False, return_bbox_im=False)
            else:
                image, target, bbox = self._getimageitem(frame, return_prompt=False, return_calib=False, return_index=False, return_bbox_im=True)
                bboxes.append(bbox)
                if self.if_last_frame_trajectory and frame_i==self.clip_length-1:
                    traj = self._draw_trajectory(target)
                    bboxes.append(traj)
            images.append(image)
            targets.append(target)
        
        self.revert_setting()
        self.set_if_return_bbox_im(if_return_bbox_im_cp)

        images = torch.stack(images)
        ret = (images, targets, )
        if return_prompt or self.if_return_prompt:
            ret += (prompt, )
        if return_calib or self.if_return_calib:
            ret += (None, )
        if return_index or self.if_return_index:
            ret += (index, )
        if return_bbox_im or self.if_return_bbox_im:
            bboxes = torch.stack(bboxes)
            ret += (bboxes, )
        return ret
    
    def _parse_label(self, label_file, frame_id):
        
        target, bbox_count = [], 0
        with open(label_file, 'r') as f:
            label = json.load(f)
            frame_i = int(frame_id[-11:-4])-1
            assert frame_id == label[frame_i]['name']
            all_ids = [int(obj['id']) for obj in label[frame_i-1]['labels']]
            for obj in label[frame_i-1]['labels']:
                if obj['category'] not in BDD100KDataset.CLASS_IDS_LOOKUP:
                    continue
                target.append({
                    'frame': frame_id,
                    'trackID': int(obj['id']),
                    'type': obj['category'],
                    'truncated': float(obj['attributes']['truncated']),
                    'occluded': int(obj['attributes']['occluded']),
                    'alpha': 0,
                    'bbox': [obj['box2d']['x1'], obj['box2d']['y1'], obj['box2d']['x2'], obj['box2d']['y2']],
                    'dimensions': [0, 0, 0],
                    'location': [0, 0, 0],
                    'rotation_y': 0,
                    'id_type': BDD100KDataset.CLASS_IDS_LOOKUP[obj['category']]
                })
                bbox_count += 1
                if bbox_count >= self.MAX_BOXES_PER_DATA:
                    # print("Exceeded max bboxes", self.MAX_BOXES_PER_DATA)
                    break
        return target

    
    def _parse_calib(self, *args):
        raise NotImplementedError('BDD100K does not have calibration files')
    
    def get_bbox_image_file_by_index(self, index=None, image_file=None):
        if image_file is None:
            image_file = self.get_image_file_by_index(index)
        if self.use_segmentation:
            return image_file.replace(BDD100KDataset.TO_IMAGE_DIR, BDD100KDataset.TO_SEG_LABELS)[:-4]+'.png'
        return image_file.replace(BDD100KDataset.TO_IMAGE_DIR, BDD100KDataset.TO_BBOX_DIR)
    
    def prompt_engineer(self, *args):
        return  'This is a real-world driving scene.'
    
    def get_image_file_by_index(self, index):
        index += 1
        image_counter = 0
        clip_folder_counter = 0
        while image_counter + self.clip_folder_lengths[self.clip_folders[clip_folder_counter]] < index:
            image_counter += self.clip_folder_lengths[self.clip_folders[clip_folder_counter]]
            clip_folder_counter += 1
        return os.path.join(self.image_dir, self.clip_folders[clip_folder_counter],
                            f'{self.clip_folders[clip_folder_counter]}-{index - image_counter:07d}.jpg')
    
    def get_clip_frame_indices(self, index):
        # this method return the same list as self.clip_list[index]
        if self.train:
            index += 1
            
            clip_folder_counter = 0
            clip_counter = 0
            while clip_counter + (self.clip_folder_lengths[self.clip_folders[clip_folder_counter]]-self.clip_length+1) < index:
                clip_counter += self.clip_folder_lengths[self.clip_folders[clip_folder_counter]]-self.clip_length+1
                clip_folder_counter += 1
            start_frame = index - clip_counter - 1
            start_frame = clip_counter + clip_folder_counter * (self.clip_length-1) + start_frame
            return list(range(start_frame, start_frame+self.clip_length))
        else:
            clip_folder_counter = 0
            frames_counter = 0
            clip_counter = 0
            while clip_counter + (self.clip_folder_lengths[self.clip_folders[clip_folder_counter]]//self.clip_length) < index+1:
                clip_counter += self.clip_folder_lengths[self.clip_folders[clip_folder_counter]]//self.clip_length
                frames_counter += self.clip_folder_lengths[self.clip_folders[clip_folder_counter]]
                clip_folder_counter += 1
                
            start_frame = frames_counter + (index - clip_counter) * self.clip_length
            return list(range(start_frame, start_frame+self.clip_length))

    def get_frame_file_by_index(self, index, timestep=None):
        if self.train:
            index += 1
            clip_folder_counter = 0
            clip_counter = 0
            while clip_counter + (self.clip_folder_lengths[self.clip_folders[clip_folder_counter]]-self.clip_length+1) < index:
                clip_counter += self.clip_folder_lengths[self.clip_folders[clip_folder_counter]]-self.clip_length+1
                clip_folder_counter += 1
            
            start_frame = index - clip_counter
            if timestep is None:
                ret = []
                for i in range(self.clip_length):
                    ret.append(os.path.join(self.image_dir, self.clip_folders[clip_folder_counter], f'{start_frame+i:07d}.jpg'))
                return ret
            assert timestep < self.clip_length
            curr_frame = start_frame+timestep
            return os.path.join(self.image_dir, self.clip_folders[clip_folder_counter],
                                f'{self.clip_folders[clip_folder_counter]}-{curr_frame:07d}.jpg')
        else:
            indices = self.get_clip_frame_indices(index)
            if timestep is None:
                ret = []
                for i in range(self.clip_length):
                    ret.append(self.get_image_file_by_index(indices[i]))
                return ret
            else:
                return self.get_image_file_by_index(indices[timestep])



    def __len__(self):
        if self.data_type == 'image':
            return sum(self.clip_folder_lengths.values())
        else:
            if self.train:
                return sum(self.clip_folder_lengths.values())+(1-self.clip_length)*len(self.clip_folder_lengths)
            else:
                counter = 0
                for _, v in self.clip_folder_lengths.items():
                    counter += v//self.clip_length
                return counter


if __name__ == "__init__":
    dataset = BDD100KDataset()
