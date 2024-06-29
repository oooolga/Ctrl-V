from .kitti_abstract import KittiAbstract

from PIL import Image
from torchvision.io import read_image
from torchvision.ops import masks_to_boxes
from torchvision import transforms as T
import torch
import os

class DAVISDataset(KittiAbstract):
    TRAIN_SPLIT = 'ImageSets/2017/train.txt'
    VAL_SPLIT = 'ImageSets/2017/val.txt'

    TO_IMAGE_DIR = 'JPEGImages/480p'
    TO_LABEL_DIR = 'Annotations_unsupervised/480p'

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
                 use_preplotted_bbox=True):

        super(DAVISDataset, self).__init__(root=root,
                                    train=train,
                                    target_transform=target_transform,
                                    data_type=data_type,
                                    clip_length=clip_length,
                                    if_return_prompt=if_return_prompt,
                                    if_return_index=if_return_index,
                                    if_return_calib=if_return_calib,
                                    if_return_bbox_im=if_return_bbox_im,
                                    H=480 if H is None else H,
                                    W=854 if W is None else W,
                                    train_H=train_H, train_W=train_W)
        
        self.version = 'davis'

        # get split ids
        if self.train:
            with open(os.path.join(self.root, self.version, DAVISDataset.TRAIN_SPLIT), 'r') as f:
                self.clip_folders = f.read().splitlines()
        else:
            with open(os.path.join(self.root, self.version, DAVISDataset.VAL_SPLIT), 'r') as f:
                self.clip_folders = f.read().splitlines()
        
        self.IDS_CLASS_LOOKUP = {i+1:_id for i, _id in enumerate(self.clip_folders)}
        self.CLASS_IDS_LOOKUP = {v:k for k, v in self.IDS_CLASS_LOOKUP.items()}

        self.clip_folder_lengths = {i:len(os.listdir(os.path.join(self.root, self.version, DAVISDataset.TO_IMAGE_DIR, i))) for i in self.clip_folders}

        if self.data_type == 'clip':
            for l in self.clip_folder_lengths.values():
                assert l >= self.clip_length, 'clip_length is larger than the number of frames in the folder'
    
    def _getimageitem(self, index, return_prompt=False, return_calib=False, return_index=False, return_bbox_im=False):
        # get the image
        image_file = self.get_image_file_by_index(index)
        image = Image.open(image_file)
        segmentation_file = self.get_bbox_image_file_by_index(image_file=image_file)
        segmentation_image = Image.open(segmentation_file)

        segmentation_tensor = T.ToTensor()(segmentation_image)
        target = self._parse_label(image_file, segmentation_tensor)

        if not self.transforms is None:
            image, target = self.transforms(image, target)

        ret = (image, target, )
        if return_prompt or self.if_return_prompt:
            prompt = self.prompt_engineer(index, image_file)
            ret += (prompt, )
        
        if return_calib or self.if_return_calib:
            ret += (None, )
        
        if return_index or self.if_return_index:
            ret += (index, )

        if return_bbox_im or self.if_return_bbox_im:
            segmentation_image = segmentation_image.convert('RGB')
            if not self.transform is None:
                segmentation_image = self.transform(segmentation_image)
            ret += (segmentation_image, )

        return ret

    def _getclipitem(self, index, return_prompt=False, return_calib=False, return_index=False, return_bbox_im=False):
        frames = self.get_clip_frame_indices(index)
        images = []
        targets = []
        bboxes = []
        
        if_return_bbox_im_cp = self.if_return_bbox_im
        self.set_if_return_bbox_im(False)
        if not self.if_return_bbox_im:
            prompt = self.prompt_engineer(frames[0])
        
        self.disable_all_settings()
        for frame in frames:
            if not (if_return_bbox_im_cp or return_bbox_im):
                image, target = self._getimageitem(frame, return_prompt=False, return_calib=False, return_index=False, return_bbox_im=False)
            else:
                image, target, bbox = self._getimageitem(frame, return_prompt=False, return_calib=False, return_index=False, return_bbox_im=True)
                bboxes.append(bbox)
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
        
    def _parse_label(self, image_file, segmentation_tensor):

        target, bbox_count = [], 0
        frame_id = int(image_file[image_file.rfind('/')+1:-4])
        object = self.get_object_by_index_or_image_file(index=None, image_file=image_file)
        try:
            boxes = masks_to_boxes(segmentation_tensor)

            for box in boxes:
                target.append({
                        'frame': frame_id,
                        'trackID': 1,
                        'type': object,
                        'truncated': float(False),
                        'occluded': int(False),
                        'alpha': 0,
                        'bbox': box.tolist(),
                        'dimensions': [0, 0, 0],
                        'location': [0, 0, 0],
                        'rotation_y': 0,
                        'id_type': self.CLASS_IDS_LOOKUP[object]
                    })
                bbox_count += 1
                if bbox_count >= self.MAX_BOXES_PER_DATA:
                    break
        except:
            pass
        return target

    def _parse_calib(self, *args):
        raise NotImplementedError('DAVIS dataset does not have calibration information')
    
    def get_object_by_index_or_image_file(self, index, image_file=None):
        if image_file is None:
            image_file = self.get_image_file_by_index(index)
        object = image_file[:image_file.rfind('/')]
        object = object[object.rfind('/')+1:]
        return object

            
    def prompt_engineer(self, index, image_file=None):
        object = self.get_object_by_index_or_image_file(index, image_file)
        return 'This is a video clip of a {}.'.format(object)
    
    def get_image_file_by_index(self, index):
        index += 1
        image_counter = 0
        clip_folder_counter = 0
        while image_counter + self.clip_folder_lengths[self.clip_folders[clip_folder_counter]] < index:
            image_counter += self.clip_folder_lengths[self.clip_folders[clip_folder_counter]]
            clip_folder_counter += 1

        return os.path.join(self.root, self.version, DAVISDataset.TO_IMAGE_DIR,
                            self.clip_folders[clip_folder_counter],
                            f'{index-image_counter-1:05d}.jpg')
    
    def get_bbox_image_file_by_index(self, index=None, image_file=None):
        if image_file is None:
            image_file = self.get_image_file_by_index(index)
        return image_file.replace(DAVISDataset.TO_IMAGE_DIR, DAVISDataset.TO_LABEL_DIR).replace('jpg', 'png')

    def get_clip_frame_indices(self, index):
        # this method return the same list as self.clip_list[index]
        index += 1
        
        clip_folder_counter = 0
        clip_counter = 0
        while clip_counter + (self.clip_folder_lengths[self.clip_folders[clip_folder_counter]]-self.clip_length+1) < index:
            clip_counter += self.clip_folder_lengths[self.clip_folders[clip_folder_counter]]-self.clip_length+1
            clip_folder_counter += 1
        start_frame = index - clip_counter - 1
        start_frame = clip_counter + clip_folder_counter * (self.clip_length-1) + start_frame
        return list(range(start_frame, start_frame+self.clip_length))
    
    def get_frame_file_by_index(self, index, timestep=None):
        indices = self.get_clip_frame_indices(index)
        if timestep is None:
            ret = []
            for i in range(self.clip_length):
                ret.append(self.get_image_file_by_index(indices[i]))
            return ret
        else:
            return self.get_image_file_by_index(indices[timestep])
    
    def __len__(self):
        return sum(self.clip_folder_lengths.values()) if self.data_type == 'image' \
            else sum(self.clip_folder_lengths.values())+(1-self.clip_length)*len(self.clip_folder_lengths)