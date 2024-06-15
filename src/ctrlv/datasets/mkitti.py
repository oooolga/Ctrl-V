from .kitti_abstract import KittiAbstract
from .kitti import KittiDataset
from .vkitti import VKittiDataset

import os
MAX_BOXES_PER_DATA = 77

class MergedKittiDataset(KittiAbstract):
    def __init__(self,
                 root="./datasets",
                 vkitti_version='vkitti_2.0.3',
                 kitti_version='kitti',
                 train=True,
                 target_transform=None,
                 data_type="image",
                 clip_length=None,
                 if_return_prompt=True,
                 if_return_index=True,
                 if_return_calib=False,
                 if_return_bbox_im=False,
                 train_H=None, train_W=None,
                 use_preplotted_bbox=True):
        super(MergedKittiDataset, self).__init__(root=root, 
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
        self.vkitti_version = vkitti_version
        self.kitti_version = kitti_version
        self.vkitti_dset = VKittiDataset(root=root, version=vkitti_version, train=train, target_transform=target_transform, data_type=data_type, clip_length=clip_length,
                                         if_return_prompt=if_return_prompt, if_return_index=False, if_return_calib=if_return_calib, if_return_bbox_im=if_return_bbox_im)
        self.kitti_dset = KittiDataset(root=root, version=kitti_version, train=train, target_transform=target_transform, data_type=data_type, clip_length=clip_length,
                                       if_return_prompt=if_return_prompt, if_return_index=False, if_return_calib=if_return_calib, if_return_bbox_im=if_return_bbox_im)
        self.images, self.image_list = None, self.vkitti_dset.image_list+self.kitti_dset.image_list

    def __len__(self):
        return len(self.vkitti_dset)+len(self.kitti_dset)

    def __getitem__(self, index,  return_prompt=False, return_calib=False, return_index=False, return_bbox_im=False):
        if index < len(self.vkitti_dset):
            ret = self.vkitti_dset.__getitem__(index, return_prompt=return_prompt, return_calib=return_calib, return_index=False, return_bbox_im=return_bbox_im)
        else:
            ret = self.kitti_dset.__getitem__(index-len(self.vkitti_dset), return_prompt=return_prompt, return_calib=return_calib, return_index=False, return_bbox_im=return_bbox_im)
        
        if self.if_return_index or return_index:
            if return_bbox_im or self.if_return_bbox_im:
                tmp = ret[-1]
                ret = ret[:-1] + (index, tmp, )
            else:
                ret += (index,)
        return ret
    
    def get_frame_file_by_index(self, index, timestep=None):
        if index < len(self.vkitti_dset):
            return self.vkitti_dset.get_frame_file_by_index(index, timestep=timestep)
        else:
            return self.kitti_dset.get_frame_file_by_index(index-len(self.vkitti_dset), timestep=timestep)
    
    def get_bbox_image_file_by_index(self, image_file):
        if 'vkitti' in image_file:
            return self.vkitti_dset.get_bbox_image_file_by_index(image_file=image_file)
        else:
            return self.kitti_dset.get_bbox_image_file_by_index(image_file=image_file)