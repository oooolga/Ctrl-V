from cv2 import cvtColor
import cv2
import wandb
import numpy as np
from . import rescale_bbox
from torchvision.utils import draw_bounding_boxes
from collections import defaultdict
from typing import List

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

def plot_trajectory(img, labels, channel_first=True, rgb2bgr=False):
    if channel_first:
        img = img.permute((1, 2, 0)).detach().cpu().numpy().copy()*255
    else:
        img = img.detach().cpu().numpy().copy()*255
    
    if rgb2bgr:
        img = cvtColor(img, cv2.COLOR_RGB2BGR)

    for i, label_info in enumerate(labels):
        type_color_i = CVCOLORS.TYPE_LOOKUP[label_info['id_type']]
        track_color_i = CVCOLORS.REVERT_CHANNEL_F(CVCOLORS.TRACKID_LOOKUP[label_info['trackID']])
        box_2d = label_info['bbox']
        x = int((box_2d[0] + box_2d[2]) / 2)
        y = int((box_2d[1] + box_2d[3]) / 2)
        
        cv2.circle(img, (x, y), 20, track_color_i, -1)
        cv2.circle(img, (x, y), 10, type_color_i, -1)
    return img
    


def plot_3d_bbox(img, labels, cam_to_img, is_gt=True, channel_first=True, rgb2bgr=False, plot_2d_bbox=False, alpha_2dbbox=0.75, box_color=None):

    if channel_first:
        img = img.permute((1, 2, 0)).detach().cpu().numpy().copy()*255
    else:
        img = img.detach().cpu().numpy().copy()*255
    
    if rgb2bgr:
        img = cvtColor(img, cv2.COLOR_RGB2BGR)
    
    bbox_3d_canvas = np.zeros_like(img)
    bbox_2d_canvas = np.zeros_like(img)
    
    for i, label_info in enumerate(labels):
        type_color_i = CVCOLORS.TYPE_LOOKUP[label_info['id_type']] if box_color is None else box_color
        if not cam_to_img is None:
            alpha = label_info['alpha']
            box_3d = []
            center = label_info['location']
            dims = label_info['dimensions']

            if is_gt:
                rot_y = label_info['rotation_y']
            else:
                rot_y = alpha / 180 * np.pi + np.arctan(center[0] / center[2])

            for i in [1, -1]:
                for j in [1, -1]:
                    for k in [0, 1]:
                        point = np.copy(center)
                        point[0] = center[0] + i * dims[1] / 2 * np.cos(-rot_y + np.pi / 2) + (j * i) * dims[2] / 2 * np.cos(
                            -rot_y)
                        point[2] = center[2] + i * dims[1] / 2 * np.sin(-rot_y + np.pi / 2) + (j * i) * dims[2] / 2 * np.sin(
                            -rot_y)
                        point[1] = center[1] - k * dims[0]
                        if cam_to_img.shape[1] == 4:
                            point = np.append(point, 1)
                        point = np.dot(cam_to_img, point)
                        point = point[:2] / point[2] if np.abs(point[2]) > 1e-4 else point[:2]/1e-4
                        point = point.astype(np.int16)
                        box_3d.append(point)
            for i in range(4):
                point_1_ = box_3d[2 * i]
                point_2_ = box_3d[2 * i + 1]
                cv2.line(bbox_3d_canvas, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), type_color_i, 2)

            for i in range(8):
                point_1_ = box_3d[i]
                point_2_ = box_3d[(i + 2) % 8]
                cv2.line(bbox_3d_canvas, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), type_color_i, 2)
            
            # draw x mark at the back of the object
            back_mark = [(box_3d[i][0], box_3d[i][1]) for i in [2, 3, 4, 5]]
            cv2.line(bbox_3d_canvas, back_mark[0], back_mark[3], type_color_i, 1)
            cv2.line(bbox_3d_canvas, back_mark[1], back_mark[2], type_color_i, 1)

        if plot_2d_bbox:
            track_color_i = CVCOLORS.REVERT_CHANNEL_F(CVCOLORS.TRACKID_LOOKUP[label_info['trackID']])
            box_2d = label_info['bbox']
            cv2.rectangle(bbox_2d_canvas, (int(box_2d[0]), int(box_2d[1])), (int(box_2d[2]), int(box_2d[3])), track_color_i, cv2.FILLED)
            if cam_to_img is None:
                cv2.rectangle(img, (int(box_2d[0]), int(box_2d[1])), (int(box_2d[2]), int(box_2d[3])), type_color_i, 2)
            # mask = shapes.astype(bool)
            # img[mask] = cv2.addWeighted(shapes, alpha_2dbbox, img, 1-alpha_2dbbox, 0)[mask]
    
    mask = bbox_2d_canvas.astype(bool)
    img[mask] = cv2.addWeighted(bbox_2d_canvas, alpha_2dbbox, img, 1-alpha_2dbbox, 0)[mask]
    mask = bbox_3d_canvas.astype(bool)
    img[mask] = bbox_3d_canvas[mask]
    return img

def save_image(image, file_path, verbose=False):
    """
    Save a numpy array as an image file.

    Args:
        image (numpy.ndarray): The input image.
        file_path (str): The file path to save the image.

    Returns:
        None
    """
    cv2.imwrite(file_path, image)
    if verbose:
        print("Image has been saved to {}".format(file_path))

def save_mask(mask, file_path, verbose=False):
    from torchvision.utils import save_image
    save_image(mask, file_path)
    if verbose:
        print("Mask has been saved to {}".format(file_path))

def tensor2wandbimage(frame_tensor, bbox_tensor=None, track_ids=None, caption=None):
    if not bbox_tensor is None:
        assert not track_ids is None
        bbox_info = []
        for bbox, track_id in zip(bbox_tensor, track_ids):
            bbox_info.append({
                'position': {
                    'minX': bbox[0].item(),
                    'minY': bbox[1].item(),
                    'maxX': bbox[2].item(),
                    'maxY': bbox[3].item()
                },
                'class_id': track_id.item(),
            })
        return wandb.Image(frame_tensor, boxes={'ground__truth': {'box_data': bbox_info}}, caption=caption)
    else:
        return wandb.Image(frame_tensor, caption=caption)
    

def wandb_frames_with_bbox(video_tensor, objects=None, image_size=(1242, 375)):
    if not objects is None:
        bboxes = objects['bbox']
        track_ids = objects['track_id'][0]
        # resize bbox to [0, 1]
        bboxes = rescale_bbox(bboxes, image_size=image_size, target_size=(1, 1))[0]
    video_tensor = np.transpose(video_tensor, (0, 2, 3, 1))
    ret = []
    num_frames = video_tensor.shape[0]
    for f_i  in range(num_frames):
        frame = video_tensor[f_i]
        bboxes_frame = bboxes[f_i] if not objects is None else None
        track_ids_frame = track_ids[f_i] if not objects is None else None
        ret.append(tensor2wandbimage(frame, bboxes_frame, track_ids_frame, caption="Frame {}".format(f_i)))
    return ret

def export_to_video(video_frames: List[np.ndarray], output_video_path: str = None, fps: int=5) -> str:
    import cv2, tempfile

    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, c = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i].astype(np.uint8), cv2.COLOR_RGB2BGR)
        video_writer.write(img)
    return output_video_path
