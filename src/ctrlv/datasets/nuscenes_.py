from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import box_in_image, view_points

import numpy as np
from pyquaternion import Quaternion
from PIL import Image
from collections import defaultdict
import cv2
import os
import torch
from ctrlv.datasets.kitti_abstract import KittiAbstract
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple

class CVCOLORS:
    RED = (0,0,255)
    GREEN = (0,255,0)
    BLUE = (255,0,0)
    PURPLE = (247,44,200)
    ORANGE = (44,162,247)
    MINT = (239,255,66)
    YELLOW = (2,255,250)
    BROWN = (42,42,165)
    LIME=(51,255,153)
    GRAY=(128, 128, 128)
    LIGHTPINK = (222,209,255)
    LIGHTGREEN = (204,255,204)
    LIGHTBLUE = (255,235,207)
    LIGHTPURPLE = (255,153,204)
    LIGHTRED = (204,204,255)
    WHITE = (255,255,255)
    BLACK = (0,0,0)
    
    TRACKID_LOOKUP = defaultdict(lambda: (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255)))
    TYPE_LOOKUP = [BLUE, WHITE, RED, YELLOW, PURPLE, BROWN, GREEN, ORANGE, LIGHTPURPLE, LIGHTRED, GRAY]
    REVERT_CHANNEL_F = lambda x: (x[2], x[1], x[0])

def render_box_3d_style(box, axis, view: np.ndarray = np.eye(3), normalize: bool = False, outline_color=(255, 0, 0), fill_color=(0, 255, 0), linewidth: float = 2, show_3d_bboxes=True, show_2d_bboxes=False) -> None:
        """
        Renders the box in the provided Matplotlib axis.
        :param axis: Axis onto which the box should be drawn.
        :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
            back and sides.
        :param linewidth: Width in pixel of the box sides.
        """
        corners = view_points(box.corners(), view, normalize=normalize)[:2, :]

        if show_3d_bboxes:
            def draw_rect(selected_corners, color):
                prev = selected_corners[-1]
                for corner in selected_corners:
                    axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth)
                    prev = corner

            # Draw the sides
            for i in range(4):
                axis.plot([corners.T[i][0], corners.T[i + 4][0]],
                          [corners.T[i][1], corners.T[i + 4][1]],
                          color=outline_color, linewidth=linewidth)

            # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
            draw_rect(corners.T[:4], outline_color)
            draw_rect(corners.T[4:], outline_color)
            
            # Draw x mark at the back of the object
            axis.plot([corners.T[4][0], corners.T[6][0], corners.T[5][0], corners.T[7][0]],
                    [corners.T[4][1], corners.T[6][1], corners.T[5][1], corners.T[7][1]],
                    color=outline_color, linewidth=1)

        if show_2d_bboxes:
            # Calculate the bottom left corner of the rectangle
            bottom_left_x, top_right_x = np.min(corners.T[:, 0]), np.max(corners.T[:, 0])
            bottom_left_y, top_right_y = np.min(corners.T[:, 1]), np.max(corners.T[:, 1])
            width, height = np.abs(bottom_left_x - top_right_x), np.abs(bottom_left_y - top_right_y)

            # Create the rectangle
            edgecolor = 'none' if show_3d_bboxes else outline_color
            rectangle = patches.Rectangle((bottom_left_x, bottom_left_y), width, height, linewidth=2, edgecolor=edgecolor, facecolor=fill_color, alpha=0.75)
            axis.add_patch(rectangle)
            
        
def my_render_3d_style(ax, nusc, boxes_3d, camera_sensor, ego_pose, data_path=None, transform=True, background=False, show_3d_bboxes=True, show_2d_bboxes=False) -> None:
    """
    Bboxes are to be in global coordinate frame, and will be projected to the specified camera
    """

    # Plot CAMERA view.
    im_size = (1600, 900) if data_path is None else Image.open(data_path).size
    ax.set_xlim(0, im_size[0])
    ax.set_ylim(im_size[1], 0)
    ax.set_aspect('equal')

    if background:
        im = Image.new('RGB', im_size) if not data_path else Image.open(data_path)
        ax.imshow(im)
        ax.axis('off')
    
    # Camera extrinsic and intrinsic parameters
    camera_intrinsic = np.array(camera_sensor['camera_intrinsic'])

    for ind, box_3d in enumerate(boxes_3d):
        
        if transform:
            # Move box to ego vehicle coord system.
            box_3d.translate(-np.array(ego_pose['translation']))
            box_3d.rotate(Quaternion(ego_pose['rotation']).inverse)
            #  Move box to sensor coord system.
            box_3d.translate(-np.array(camera_sensor['translation']))
            box_3d.rotate(Quaternion(camera_sensor['rotation']).inverse)

        # Only render bboxes that fit in image frame
        if not box_in_image(box_3d, camera_intrinsic, im_size, vis_level=1):
            continue
            
        outline_color = np.array(CVCOLORS.REVERT_CHANNEL_F(CVCOLORS.TYPE_LOOKUP[NuScenesDataset.NUSC_CLASS_TO_GROUP_IDS[box_3d.name]])) / 255.0
        # fill_color = np.array(CVCOLORS.REVERT_CHANNEL_F(CVCOLORS.TRACKID_LOOKUP[box_3d.token])) / 255.0
        annotation_data = nusc.get('sample_annotation', box_3d.token)
        instance_token = annotation_data['instance_token']
        fill_color = np.array(CVCOLORS.REVERT_CHANNEL_F(CVCOLORS.TRACKID_LOOKUP[instance_token])) / 255.0
        
        render_box_3d_style(box_3d, ax, view=camera_intrinsic, normalize=True, outline_color=outline_color, fill_color=fill_color, show_3d_bboxes=show_3d_bboxes, show_2d_bboxes=show_2d_bboxes)

class NuScenesDataset(KittiAbstract):

    # Based on closest match to KITTI classes
    NUSC_CLASS_TO_GROUP_IDS_KITTI = {
        "human.pedestrian.adult": 4,
        "human.pedestrian.child":  4,
        "human.pedestrian.construction_worker": 5,
        "human.pedestrian.personal_mobility": 4,
        "human.pedestrian.police_officer": 5,
        "movable_object.barrier": 8,
        "movable_object.debris":  8,
        "movable_object.pushable_pullable": 8,
        "movable_object.trafficcone": 8,
        "static_object.bicycle_rack":  8,
        "vehicle.bicycle":  6,
        "vehicle.bus.bendy":  3,
        "vehicle.bus.rigid":  3,
        "vehicle.car": 1,
        "vehicle.construction":  3,
        "vehicle.motorcycle":  6,  # NOTE: Not sure if best to classify as cyclist or car...
        "vehicle.trailer":  3,
        "vehicle.truck":  3,
    }
    # Based on closest match to BDD100k classes
    NUSC_CLASS_TO_GROUP_IDS = NUSC_CLASS_TO_GROUP_IDS_BDD = {
        "human.pedestrian.adult": 1,
        "human.pedestrian.child":  1,
        "human.pedestrian.construction_worker": 1,
        "human.pedestrian.personal_mobility": 1,
        "human.pedestrian.police_officer": 1,
        "movable_object.barrier": 10,
        "movable_object.debris":  10,
        "movable_object.pushable_pullable": 10,
        "movable_object.trafficcone": 10,
        "static_object.bicycle_rack":  10,
        "vehicle.bicycle":  8,
        "vehicle.bus.bendy":  5,
        "vehicle.bus.rigid":  5,
        "vehicle.car": 3,
        "vehicle.construction":  4,
        "vehicle.motorcycle":  7, 
        "vehicle.trailer":  4,
        "vehicle.truck":  4,
    }

    def __init__(self,
                 root='/network/scratch/a/anthony.gosselin', 
                 bbox_dir=None,
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
                 use_preplotted_bbox=True,
                 if_3d=False):

        super(NuScenesDataset, self).__init__(root=root, 
                                              train=train, 
                                              target_transform=target_transform, 
                                              data_type=data_type, 
                                              clip_length=clip_length,
                                              if_return_prompt=if_return_prompt,
                                              if_return_index=if_return_index,
                                              if_return_calib=if_return_calib,
                                              if_return_bbox_im=if_return_bbox_im,
                                              H=900 if H is None else H,
                                              W=1600 if W is None else W,
                                              train_H=train_H, train_W=train_W,
                                              use_preplotted_bbox=use_preplotted_bbox)
        
        # assert data_type == 'clip', "Only clip data type is supported for NuScenes dataset."
        self.version = 'nuscenes'
        self.nusc = NuScenes(version='v1.0-trainval', # 'v1.0-mini'
                             dataroot=os.path.join(self.root, self.version),
                             verbose=True)
        self.if_3d = if_3d

        self.scene_sample_dict = defaultdict(dict)
        self.idx_sample_dict = []
        self.bbox_dir = bbox_dir
        
        self.TRACKID_LOOKUP = {}

        self.fps = 7 # NOTE: when setting to fps=7 with -0.05 correction term on target_period, the real fps is more like 8
        target_period = 1/self.fps - 0.05  # For fps downsampling
        for scene_i, scene in enumerate(self.nusc.scene):
            
            self.scene_sample_dict[scene['name']]['scene'] = scene_i
            self.scene_sample_dict[scene['name']]['frontcam_samples'] = []  # NOTE: Currently only using front camera data
            
            curr_data_token = self.nusc.get('sample', scene['first_sample_token'])['data']["CAM_FRONT"]
            self.scene_sample_dict[scene['name']]['frontcam_samples'].append(curr_data_token)
            
            cumul_delta = 0
            while curr_data_token:
                curr_sample_data = self.nusc.get('sample_data', curr_data_token)
                
                next_sample_data_token = curr_sample_data['next']
                if not next_sample_data_token:
                    break
                next_sample_data = self.nusc.get('sample_data', next_sample_data_token)

                # FPS downsampling: only select certain frames based on elapsed times 
                delta = (next_sample_data['timestamp'] - curr_sample_data['timestamp']) / 1e6
                cumul_delta += delta
                if cumul_delta >= target_period:
                    self.scene_sample_dict[scene['name']]['frontcam_samples'].append(next_sample_data_token)
                    cumul_delta = 0

                curr_data_token = next_sample_data_token
            
            if self.data_type == 'image':
                for i in range(len(self.scene_sample_dict[scene['name']]['frontcam_samples'])):
                    self.idx_sample_dict.append(self.scene_sample_dict[scene['name']]['frontcam_samples'][i])
            
            elif self.data_type == 'clip':
                # In case self.clip_length << actual video sample length (~20s), we can create multiple non-overlapping clips for each sample
                total_frames = len(self.scene_sample_dict[scene['name']]['frontcam_samples'])
                for clip_i in range(total_frames // self.clip_length):
                    start_image_idx = clip_i * self.clip_length
                    self.idx_sample_dict.append(self.scene_sample_dict[scene['name']]['frontcam_samples'][start_image_idx])
        
    def __len__(self):
        return len(self.idx_sample_dict)
    
    def _getimageitem(self, index, token=None, return_prompt=False, return_calib=False, return_index=False, return_bbox_im=False):
        if token is None:
            token = self.idx_sample_dict[index]

        data = self.nusc.get('sample_data', token)
        image_path = os.path.join(self.root, self.version, data['filename'])
        image = Image.open(image_path)

        target = self._parse_label(token)

        if not self.transforms is None:
            image, target = self.transforms(image, target)

        ret = (image, target)
        if return_prompt or self.if_return_prompt:
            prompt = self.prompt_engineer()
            ret += (prompt, )
        if return_calib or self.if_return_calib:
            # get the camera calibration information
            ret += (None, )
        
        if return_index or self.if_return_index:
            ret += (index, )
        
        if return_bbox_im or self.if_return_bbox_im:

            fig_path = os.path.join(self.bbox_dir, f'{token}.png') if self.bbox_dir is not None else 'temp.png'

            if self.bbox_dir is None or not os.path.exists(fig_path):
                bboxes = self.nusc.get_boxes(token)

                sample_data = self.nusc.get('sample_data', token)
                ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
                front_camera_sensor = self.nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])

                fig, ax = plt.subplots()
                bbox_im = np.zeros((self.orig_W, self.orig_H, 3), dtype=np.uint8)
                my_render_3d_style(ax, self.nusc, bboxes, front_camera_sensor, ego_pose, None, background=True, show_2d_bboxes=True, show_3d_bboxes=self.if_3d)
                ax.axis('off')
                plt.margins(x=0, y=0)

                # Convert plot to image
                fig.tight_layout()
                fig.savefig(fig_path,  bbox_inches='tight', pad_inches=0)
                plt.clf()
            
            img = Image.open(fig_path).convert('RGB')
            bbox_im = self.transform(img)

            ret += (bbox_im, )
        return ret
    
    def _getclipitem(self, index, return_prompt=False, return_calib=False, return_index=False, return_bbox_im=False):
        curr_token = self.idx_sample_dict[index]
        images = []
        targets = []
        bboxes = []
        
        if_return_bbox_im_cp = self.if_return_bbox_im
        self.set_if_return_bbox_im(False)
        if not self.if_return_bbox_im:
            prompt = self.prompt_engineer()
        
        self.disable_all_settings()
        traj = None
        for i in range(self.clip_length):
            curr_sample_data = self.nusc.get('sample_data', curr_token)

            if not (if_return_bbox_im_cp or return_bbox_im):
                image, target = self._getimageitem(index, token=curr_token, return_prompt=False, return_calib=False, return_index=False, return_bbox_im=False)
            else:
                image, target, bbox = self._getimageitem(index, token=curr_token, return_prompt=False, return_calib=False, return_index=False, return_bbox_im=True)
                bboxes.append(bbox)
            
            images.append(image)
            targets.append(target)

            curr_token = curr_sample_data['next']
        
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
    
    def _parse_label(self, token):
        cam_front_data = self.nusc.get('sample_data', token)
        front_camera_sensor = self.nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])
        camera_intrinsic = np.array(front_camera_sensor['camera_intrinsic'])

        target = []
        for bbox_3d in self.nusc.get_boxes(token):

            if bbox_3d.name not in NuScenesDataset.NUSC_CLASS_TO_GROUP_IDS or NuScenesDataset.NUSC_CLASS_TO_GROUP_IDS_KITTI[bbox_3d.name] == 8:
                continue
            id_type = NuScenesDataset.NUSC_CLASS_TO_GROUP_IDS_KITTI[bbox_3d.name]
            
            instance_token = self.nusc.get('sample_annotation', bbox_3d.token)['instance_token']
            if instance_token not in self.TRACKID_LOOKUP:
                self.TRACKID_LOOKUP[instance_token] = len(self.TRACKID_LOOKUP)

            target.append(
                {
                    'frame': None,
                    'trackID': self.TRACKID_LOOKUP[instance_token],
                    'type': bbox_3d.name,
                    'truncated': 0,
                    'occluded': 0,
                    'alpha': bbox_3d.orientation.radians,
                    'bbox': [np.min(view_points(bbox_3d.corners(), camera_intrinsic, normalize=True)[0,:]),
                             np.min(view_points(bbox_3d.corners(), camera_intrinsic, normalize=True)[1,:]), 
                             np.max(view_points(bbox_3d.corners(), camera_intrinsic, normalize=True)[0,:]),
                             np.max(view_points(bbox_3d.corners(), camera_intrinsic, normalize=True)[1,:])], # TODO: these values are incorrect
                    'dimensions': [bbox_3d.wlh[2], bbox_3d.wlh[0], bbox_3d.wlh[1]],
                    'location': [bbox_3d.center[0], bbox_3d.center[1], bbox_3d.center[2]],
                    'rotation_y': bbox_3d.orientation.axis[1],
                    'id_type': id_type,
                }
            )
        return target
    
    def render_cv2(self,
                   bbox,
                   im: np.ndarray,
                   view: np.ndarray = np.eye(3),
                   normalize: bool = False,
                   colors: Tuple = ((0, 0, 255), (255, 0, 0), (155, 155, 155)),
                   linewidth: int = 2) -> None:
        """
        Renders box using OpenCV2.
        :param im: <np.array: width, height, 3>. Image array. Channels are in BGR order.
        :param view: <np.array: 3, 3>. Define a projection if needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: ((R, G, B), (R, G, B), (R, G, B)). Colors for front, side & rear.
        :param linewidth: Linewidth for plot.
        """
        corners = view_points(bbox.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                cv2.line(im,
                         (int(prev[0]), int(prev[1])),
                         (int(corner[0]), int(corner[1])),
                         color, linewidth)
                prev = corner

        # Draw the sides
        for i in range(4):
            cv2.line(im,
                     (int(corners.T[i][0]), int(corners.T[i][1])),
                     (int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
                     colors[2][::-1], linewidth)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0][::-1])
        draw_rect(corners.T[4:], colors[1][::-1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        cv2.line(im,
                 (int(center_bottom[0]), int(center_bottom[1])),
                 (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
                 colors[0][::-1], linewidth)
        
        return im
    
    def get_frame_file_by_index(self, index, timestep=None):
        counter = 0
        if timestep is None:
            ret = []
        
        token = self.idx_sample_dict[index]
        while True:

            sample_data = self.nusc.get('sample_data', token)
            image_path = os.path.join(self.root, self.version, sample_data['filename'])

            if timestep == counter:
                return image_path
            if timestep is None:
                ret.append(image_path)
            
            counter += 1
            if counter == self.clip_length:
                return ret
            
            token = sample_data['next']
    
    def get_bbox_image_file_by_index(self, index, timestep=0, image_file=None):
        
        orig_bbox_dir = self.bbox_dir
        if self.bbox_dir is None:
            self.bbox_dir = '.'
        
        self._getclipitem(index, return_bbox_im=True)
        counter = 0
        if timestep is None:
            ret = []
        
        token = self.idx_sample_dict[index]
        while True:

            sample_data = self.nusc.get('sample', token)
            image_path = os.path.join(self.bbox_dir, f'{token}.png')

            if timestep == counter:
                self.bbox_dir = orig_bbox_dir
                return image_path
            if timestep is None:
                ret.append(image_path)
            
            counter += 1
            if counter == self.clip_length:
                self.bbox_dir = orig_bbox_dir
                return ret
            
            token = sample_data['next']
    
    def prompt_engineer(self, *args):
        # NOTE: nuScenes has actual descriptions for each scenes (scene['description']) that can be used
        return  'This is a real-world driving scene.'

    
if __name__ == "__main__":
    nusc = NuScenesDataset(root='/network/scratch/a/anthony.gosselin/', train=True, data_type='clip', clip_length=16)
    print("Done.")