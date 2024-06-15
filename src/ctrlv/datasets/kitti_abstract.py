from torchvision.datasets.vision import VisionDataset
from torchvision import transforms
from abc import ABC
import torch
from ..utils import plot_3d_bbox, plot_trajectory

# class RevertPixelRange(torch.nn.Module):
#     def forward(self, img):
#         return img * 0.5 + 0.5

class KittiAbstract(ABC, VisionDataset):

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
    MAX_BOXES_PER_DATA = 15

    IDS_CLASS_LOOKUP = {1: 'Car',
                        2: 'Van',
                        3: 'Truck',
                        4: 'Pedestrian',
                        5: 'Person',
                        6: 'Cyclist',
                        7: 'Tram',
                        8: 'Misc',
                        9: 'DontCare'}
    
    CLASS_IDS_LOOKUP = {'Car':1,
                        'Van':2,
                        'Truck':3,
                        'Pedestrian':4,
                        'Person':5,
                        'Cyclist':6,
                        'Tram':7,
                        'Misc':8,
                        'DontCare':9
                        }
    
    TO_COCO_LABELS = {
        1: 2,
        2: 2,
        3: 7,
        4: 0,
        5: 0,
        6: 0,
        7: 6,
    }

    def __init__(self, root='./datasets',
                 train=True,
                 target_transform=None,
                 data_type='image',
                 clip_length=None,
                 if_return_prompt=True,
                 if_return_index=True,
                 if_return_calib=False,
                 if_return_bbox_im=False,
                 use_preplotted_bbox=True,
                 H=None, W=None,
                 train_H=None, train_W=None):
        
        # self.H, self.W = 375, 1242
        if H is None:
            self.orig_H = 375
        else:
            self.orig_H = H
        if W is None:
            self.orig_W = 1242
        else:
            self.orig_W = W
        if train_H is None:
            self.train_H = 320#512 if data_type == 'image' else 320
        else:
            self.train_H = train_H
        self.train_W = 512 if train_W is None else train_W

        transform=transforms.Compose([
                    transforms.Resize((self.train_H, self.train_W)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), # map from [0,1] to [-1,1]
                 ])
        self.revert_transform = transforms.Compose([
            transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),
            transforms.Resize((self.orig_H, self.orig_W)),
            ])
        self.revert_transform_no_resize = transforms.Compose([
            transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),
            ])
        super(KittiAbstract, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train
        assert data_type in ['image', 'clip']
        self.data_type = data_type
        self.clip_length = clip_length
        assert self.data_type == 'image' or not self.clip_length is None

        self.images, self.image_list = {}, []
        
        self.set_if_return_prompt(if_return_prompt)
        self.set_if_return_index(if_return_index)
        self.set_if_return_calib(if_return_calib)
        self.set_if_return_bbox_im(if_return_bbox_im)
        self.copy_setting()
        self.if_return_im = True
        self.use_preplotted_bbox = use_preplotted_bbox
        self.if_last_frame_trajectory = False

    def disable_get_image(self):
        import warnings
        warnings.warn("Disabling get image function")
        self.if_return_im = False
    
    def set_if_last_frame_trajectory(self, flag):
        self.if_last_frame_trajectory = flag

    def set_if_return_index(self, flag):
        self.if_return_index = flag
    
    def set_if_return_prompt(self, flag):
        self.if_return_prompt = flag
    
    def set_if_return_calib(self, flag):
        self.if_return_calib = flag
    
    def set_if_return_bbox_im(self, flag):
        self.if_return_bbox_im = flag
    
    def copy_setting(self):
        self.if_return_index_cp, self.if_return_prompt_cp, self.if_return_calib_cp, self.if_return_bbox_im_cp = self.if_return_index, self.if_return_prompt, self.if_return_calib, self.if_return_bbox_im
    
    def revert_setting(self):
        self.if_return_index, self.if_return_prompt, self.if_return_calib, self.if_return_bbox_im = self.if_return_index_cp, self.if_return_prompt_cp, self.if_return_calib_cp, self.if_return_bbox_im_cp
    
    def disable_all_settings(self):
        self.copy_setting()
        self.set_if_return_index(False)
        self.set_if_return_prompt(False)
        self.set_if_return_calib(False)
        self.set_if_return_bbox_im(False)
    
    def _getimageitem(self, index, return_prompt=False, return_calib=False, return_index=False):
        pass

    def __getitem__(self, index, return_prompt=False, return_calib=False, return_index=False, return_bbox_im=False):
        if self.data_type == "image":
            return self._getimageitem(index, return_prompt=return_prompt, return_calib=return_calib,
                                      return_index=return_index, return_bbox_im=return_bbox_im)
        elif self.data_type == "clip":
            return self._getclipitem(index, return_prompt=return_prompt, return_calib=return_calib,
                                      return_index=return_index, return_bbox_im=return_bbox_im)

        
    def _getclipitem(self, index, return_prompt=False, return_calib=False, return_index=False, return_bbox_im=False):
        frames = self.clip_list[index]
        images = []
        targets = []
        bboxes = []
        
        if_return_bbox_im_cp = self.if_return_bbox_im
        self.set_if_return_bbox_im(False)
        if not self.if_return_bbox_im:
            _, _, prompt, calib, frame_index = self._getimageitem(frames[0], return_prompt=True, return_calib=True, return_index=True)
        
        self.disable_all_settings()
        for frame_i, frame in enumerate(frames):
            if not (if_return_bbox_im_cp or return_bbox_im):
                image, target = self._getimageitem(frame, return_prompt=False, return_calib=False, return_index=False, return_bbox_im=False)
            else:
                if self.if_last_frame_trajectory and frame_i==self.clip_length-1:
                    image, target, bbox = self._getimageitem(frame, return_prompt=False, return_calib=False, return_index=False, return_bbox_im=True, if_traj=True)
                else:
                    image, target, bbox = self._getimageitem(frame, return_prompt=False, return_calib=False, return_index=False, return_bbox_im=True)
                bboxes.append(bbox)
            images.append(image)
            targets.append(target)
        
        self.revert_setting()
        self.set_if_return_bbox_im(if_return_bbox_im_cp)

        images = torch.stack(images)if self.if_return_im else None
        ret = (images, targets, )
        if return_prompt or self.if_return_prompt:
            ret += (prompt, )
        if return_calib or self.if_return_calib:
            ret += (calib, )
        if return_index or self.if_return_index:
            ret += (index, )
        if return_bbox_im or self.if_return_bbox_im:
            bboxes = torch.stack(bboxes)
            ret += (bboxes, )
        return ret
    
    def _create_mask_img(self, img, bboxes=[], target=[]):
        _, h, w = img.size()
        mask = torch.zeros((h, w))
        for label in target:
            if label['type'] == 'DontCare':
                continue
            bbox = label['bbox']
            mask[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])] = 1
        for bbox in bboxes:
            mask[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])] = 1
        return mask
    
    def _draw_bbox(self, objects, cam_to_img=None, plot_2d_bbox=True):
        canvas = torch.zeros((3, self.orig_H, self.orig_W))
        bbox_im = plot_3d_bbox(canvas, objects, cam_to_img, plot_2d_bbox=plot_2d_bbox)
        transform = transforms.Compose([transforms.ToPILImage()])
        bbox_im = self.transform(transform(bbox_im))
        return bbox_im
    
    def get_trajectory_image_by_index(self, index):
        assert not self.use_preplotted_bbox, "This function is only available for non-preplotted bbox"
        _, objects, _, _, _, _ = self._getimageitem(index,
                                                    return_prompt=True,
                                                    return_calib=True,
                                                    return_index=True,
                                                    return_bbox_im=True)
        return self._draw_trajectory(objects)
    
    def _draw_trajectory(self, objects):
        canvas = torch.zeros((3, self.orig_H, self.orig_W))
        traj_im = plot_trajectory(canvas, objects, channel_first=True)
        transform = transforms.Compose([transforms.ToPILImage()])
        traj_im = self.transform(transform(traj_im))
        return traj_im
    
    def prompt_engineer(self, index):
        pass

    def get_image_file_by_index(self, index):
        return self.image_list[index]
    
    def get_frame_file_by_index(self, index, timestep=None):
        frames = self.clip_list[index]
        if timestep is None:
            ret = []
            for frame in frames:
                ret.append(self.image_list[frame])
            return ret
        return self.image_list[frames[timestep]]