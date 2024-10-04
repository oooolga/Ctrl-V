from ctrlv.utils import get_dataloader, save_image, export_to_video
import torch, os
from tqdm import tqdm

import argparse
import cv2, torchvision
import numpy as np
import os
from functools import partial
from pathlib import Path
from colour import Color

def plot_3d_bbox(img, labels, cam_to_img, is_gt=True, channel_first=True, rgb2bgr=False, plot_2d_bbox=False, alpha_2dbbox=0.75, box_color=None):
    
    for i, label_info in enumerate(labels):
        type_color_i = box_color
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
                cv2.line(img, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), type_color_i, 4)

            for i in range(8):
                point_1_ = box_3d[i]
                point_2_ = box_3d[(i + 2) % 8]
                cv2.line(img, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), type_color_i, 4)
            
            # draw x mark at the back of the object
            back_mark = [(box_3d[i][0], box_3d[i][1]) for i in [2, 3, 4, 5]]
            cv2.line(img, back_mark[0], back_mark[3], type_color_i, 3)
            cv2.line(img, back_mark[1], back_mark[2], type_color_i, 3)
        
        if plot_2d_bbox:
            box_2d = label_info['bbox']
            # cv2.rectangle(bbox_2d_canvas, (int(box_2d[0]), int(box_2d[1])), (int(box_2d[2]), int(box_2d[3])), track_color_i, cv2.FILLED)
            if cam_to_img is None:
                cv2.rectangle(img, (int(box_2d[0]), int(box_2d[1])), (int(box_2d[2]), int(box_2d[3])), type_color_i, 2)

    return img

from accelerate.utils import write_basic_config
write_basic_config()
import warnings
from diffusers.utils import is_wandb_available

import logging, os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import wandb
from collections import defaultdict
from numpy import inf

import torch
torch.cuda.empty_cache()
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from ctrlv.utils import parse_args, get_dataloader, eval_samples_generator, eval_demo_samples_generator, wandb_frames_with_bbox
    from ctrlv.pipelines import VideoDiffusionPipeline, StableVideoControlPipeline

logger = get_logger(__name__, log_level="INFO")

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    accelerator.init_trackers(args.project_name, config=vars(args), init_kwargs={"wandb": {"dir": args.output_dir, "name": args.run_name}})
    wandb.define_metric("inference_step")
    plot_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    try:
        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)
        # If passed along, set the training seed now.
        if args.seed is not None:
            set_seed(args.seed)

        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

        dataset, data_loader = get_dataloader(args.data_root, args.dataset_name, if_train=True, clip_length=args.clip_length,
                                              batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers, 
                                              data_type='clip', use_default_collate=True, tokenizer=None, shuffle=True,
                                              if_return_bbox_im=True, train_H=args.train_H, train_W=args.train_W,
                                              use_segmentation=args.use_segmentation)
        sample_generator = eval_samples_generator(data_loader)

        def run_pipelines(bbox_pipeline, ctrl_pipeline, sample_generator, one_step_limit=1):
            resize = torchvision.transforms.Resize(size=(dataset.orig_H, dataset.orig_W))
            
            from ctrlv.metrics import f_measure, binary_mask_iou
            # from ignite.engine import create_supervised_evaluator

            log_dict = defaultdict(list)
            # metric_scores = defaultdict(list)

            for sample_i, sample in tqdm(enumerate(sample_generator)):
                for j in range(3):
                    bbox_pipeline.to(accelerator.device)
                    best_score = -inf
                    for min_guidance_scale, max_guidance_scale in zip([1, 1, 2, 2, 3], [2, 3, 4, 5, 5]):
                        bbox_im = bbox_pipeline(sample['image_init'], 
                                                height=dataset.train_H, width=dataset.train_W, 
                                                bbox_images=sample['bbox_img'].unsqueeze(0),
                                                decode_chunk_size=8, motion_bucket_id=127, fps=args.fps, 
                                                num_inference_steps=30,
                                                num_frames=args.clip_length,
                                                min_guidance_scale=min_guidance_scale,
                                                max_guidance_scale=max_guidance_scale,
                                                noise_aug_strength=args.noise_aug_strength,
                                                generator=generator, output_type='pt',
                                                num_cond_bbox_frames=args.num_cond_bbox_frames).frames[0]
                        
                        bbox_frames = bbox_im.detach().cpu().numpy()*255
                        tmp = bbox_frames.sum(axis=1) < 50
                        bbox_frames[np.repeat(tmp[:, None,::], 3, axis=1)] = 0
                        
                        for frame_i in range(1, bbox_frames.shape[0]-1):
                            if bbox_frames[frame_i].sum(axis=0).min() > 50:
                                # something wrong
                                bbox_frames[frame_i] = np.zeros_like(bbox_frames[frame_i])
                    
                        bbox_frames = bbox_frames.astype(np.uint8)
                        clip_miou, clip_ap, clip_ar = binary_mask_iou(sample['bbox_img_np'][:args.clip_length], bbox_frames)
                        best_score = max(best_score, clip_miou)
                        if best_score == clip_miou:
                            best_generation_bbox = bbox_im
                            best_ap = clip_ap
                            best_ar = clip_ar
                            first_and_last_miou, first_and_last_ap, first_and_last_ar = binary_mask_iou(sample['bbox_img_np'][[0,-1],::], bbox_frames[[0,-1],::])
                            best_generation_np = bbox_frames
                        del bbox_im, bbox_frames, tmp, clip_miou, clip_ar, clip_ap

                    log_dict = {}
                    log_dict[f'predicted_bbox_{sample_i}'] = wandb.Video(best_generation_np, fps=args.fps)
                    frame_bboxes = wandb_frames_with_bbox(best_generation_np, sample['objects_tensors'], (dataset.orig_W, dataset.orig_H))
                    log_dict["bbox_frames_{}".format(sample_i)] = frame_bboxes
                    del frame_bboxes
                    best_generation_np = np.moveaxis(np.transpose(best_generation_np, (0, 2, 3, 1)), 0, 0)
                    export_to_video(best_generation_np, f"{plot_dir}/predicted_bbox_{sample_i}_{j}.mp4", fps=args.fps)

                    bbox_pipeline.to('cpu')
                    del best_generation_np

                    ctrl_pipeline.to(accelerator.device)
                    frames = ctrl_pipeline(sample['image_init'], 
                                        cond_images=2*(best_generation_bbox-0.5).unsqueeze(0),
                                        height=dataset.train_H, width=dataset.train_W, 
                                        decode_chunk_size=8, motion_bucket_id=127, fps=args.fps, 
                                        num_inference_steps=args.num_inference_steps,
                                        num_frames=args.clip_length,
                                        control_condition_scale=args.conditioning_scale,
                                        min_guidance_scale=args.min_guidance_scale,
                                        max_guidance_scale=args.max_guidance_scale,
                                        noise_aug_strength=args.noise_aug_strength,
                                        generator=generator, output_type='pt').frames[0]
                    
                    ctrl_pipeline.to('cpu')
                    frames = resize(frames)
                    frames = frames.detach().cpu().numpy()*255
                    frames = frames.astype(np.uint8)
                    log_dict[f"generated_videos_{sample_i}"] = wandb.Video(frames, fps=args.fps)
                    tmp = np.moveaxis(np.transpose(frames, (0, 2, 3, 1)), 0, 0)
                    export_to_video(tmp, f"{plot_dir}/generated_videos_{sample_i}_{j}.mp4", fps=args.fps)
                    frame_bboxes = wandb_frames_with_bbox(frames, sample['objects_tensors'], (dataset.orig_W, dataset.orig_H))
                    log_dict["frames_with_bboxes_{}".format(sample_i)] = frame_bboxes

                    accelerator.log(log_dict)
                    del frames, log_dict, frame_bboxes, best_generation_bbox, tmp
                
                log_dict = {}
                
                gt_frames_bboxes = wandb_frames_with_bbox(sample['bbox_img_np'], sample['objects_tensors'], (dataset.orig_W, dataset.orig_H))
                gt_frames = wandb_frames_with_bbox(sample['gt_clip_np'], sample['objects_tensors'], (dataset.orig_W, dataset.orig_H))
                log_dict["gt_bboxes_{}".format(sample_i)] = gt_frames_bboxes
                log_dict["gt_frames_{}".format(sample_i)] = gt_frames
                log_dict["gt_bbox_frames"] = wandb.Video(sample['bbox_img_np'], fps=args.fps)
                
                tmp = np.moveaxis(np.transpose(sample['bbox_img_np'], (0, 2, 3, 1)), 0, 0)
                export_to_video(tmp, f"{plot_dir}/gt_bbox_frames_{sample_i}.mp4", fps=args.fps)
                log_dict["gt_videos"] = wandb.Video(sample['gt_clip_np'], fps=args.fps)
                tmp = np.moveaxis(np.transpose(sample['gt_clip_np'], (0, 2, 3, 1)), 0, 0)
                export_to_video(tmp, f"{plot_dir}/gt_videos_{sample_i}.mp4", fps=args.fps)

                bbox_frames = []
                for f_i in range(len(sample['gt_labels'])):
                    canvas = np.zeros((dataset.orig_H, dataset.orig_W, 3))+255
                    if f_i == 0:
                        r, g, b = Color("plum").rgb
                    else:
                        r, g, b = Color("gold").rgb
                    color = int(255*b), int(255*g), int(255*r)
                    canvas = plot_3d_bbox(canvas, sample['gt_labels'][f_i], sample['cam_to_img'], plot_2d_bbox=args.dataset_name=='bdd100k', box_color=color, channel_first=canvas.shape[0]==3)
                    bbox_frames.append(wandb.Image(np.uint(canvas)))
                log_dict["gt_3d_bbox_frames"] = bbox_frames
                accelerator.log(log_dict)
                del gt_frames_bboxes, gt_frames, log_dict, sample, tmp
                torch.cuda.empty_cache()
                if sample_i >= args.num_demo_samples:
                    break

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            with torch.autocast(
                str(accelerator.device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"
            ):
                from ctrlv.models import UNetSpatioTemporalConditionModel
                if not os.path.isfile(os.path.join(args.pretrained_bbox_model, "unet")):
                    # Get the most recent checkpoint
                    dirs = os.listdir(args.pretrained_bbox_model)
                    dirs = [d for d in dirs if d.startswith("checkpoint")]
                    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                    path = dirs[-1] if len(dirs) > 0 else None
                    args.pretrained_bbox_model = os.path.join(args.pretrained_bbox_model, path)
                bbox_unet = UNetSpatioTemporalConditionModel.from_pretrained(args.pretrained_bbox_model,
                                                                             subfolder="unet",
                                                                             low_cpu_mem_usage=True, 
                                                                             num_frames=args.clip_length)
                bbox_pipeline = VideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt",
                                                                       unet = bbox_unet,)
                bbox_pipeline = bbox_pipeline.to('cpu')

                from ctrlv.models import UNetSpatioTemporalConditionModel, ControlNetModel
                ctrlnet = ControlNetModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="controlnet")
                unet = UNetSpatioTemporalConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
                pipeline = StableVideoControlPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt",
                                                                      controlnet = ctrlnet,
                                                                      unet = unet,)
                pipeline = pipeline.to('cpu')
                result_generator = run_pipelines(bbox_pipeline, pipeline, sample_generator)
        
        logging.info("Finished evaluation.")
        accelerator.end_training()

    except KeyboardInterrupt:
        accelerator.end_training()
        if is_wandb_available():
            wandb.finish()
        print("Keboard interrupt: shutdown requested... Exiting.")
        exit()
    except Exception:
        import sys, traceback
        if is_wandb_available():
            wandb.finish()
        traceback.print_exc(file=sys.stdout)
        sys.exit(0)

if __name__ == '__main__':
    main()