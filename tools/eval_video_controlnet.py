from accelerate.utils import write_basic_config
write_basic_config()
import warnings

import logging
import os, pickle
from pathlib import Path
import numpy as np
import accelerate
from collections import defaultdict

import torch
torch.cuda.empty_cache()
import torch.utils.checkpoint
from diffusers.utils import is_wandb_available

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from ctrlv.utils import parse_args, get_dataloader, eval_samples_generator, get_n_training_samples, wandb_frames_with_bbox, export_to_video
    from ctrlv.pipelines import StableVideoControlPipeline

if not is_wandb_available():
    warnings.warn("Make sure to install wandb if you want to use it for logging during training.")
else: 
    import wandb
logger = get_logger(__name__, log_level="INFO")


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)
    save_gt_path = os.path.join(args.output_dir, 'gt_labels_ctrl_eval')
    os.makedirs(save_gt_path, exist_ok=True)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
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

        dataset, data_loader = get_dataloader(args.data_root, args.dataset_name, if_train=False, clip_length=args.clip_length,
                                              batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers, 
                                              data_type='clip', use_default_collate=True, tokenizer=None, shuffle=True,
                                              if_return_bbox_im=True, train_H=args.train_H, train_W=args.train_W,
                                              use_segmentation=args.use_segmentation)
        # sample_generator = get_n_training_samples(data_loader, args.num_demo_samples)
        sample_generator = eval_samples_generator(data_loader)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            accelerator.init_trackers(args.project_name, config=vars(args), init_kwargs={"wandb": {"dir": args.output_dir, "name": args.run_name}})


        def run_inference_with_pipeline(pipeline, demo_samples):
            gt_labels = []
            for sample_i, sample in enumerate(demo_samples):
                frames = pipeline(sample['image_init'] if not args.generate_bbox else sample['bbox_init'], 
                                cond_images=sample['bbox_img'].unsqueeze(0) if not args.generate_bbox else sample['gt_clip'].unsqueeze(0),
                                height=dataset.train_H, width=dataset.train_W, 
                                decode_chunk_size=8, motion_bucket_id=127, fps=args.fps, 
                                num_inference_steps=args.num_inference_steps,
                                num_frames=args.clip_length,
                                control_condition_scale=args.conditioning_scale,
                                min_guidance_scale=args.min_guidance_scale,
                                max_guidance_scale=args.max_guidance_scale,
                                noise_aug_strength=args.noise_aug_strength,
                                generator=generator, output_type='pt').frames[0]
                #frames = F.interpolate(frames, (dataset.orig_H, dataset.orig_W)).detach().cpu().numpy()*255
                frames = frames.detach().cpu().numpy()*255
                frames = frames.astype(np.uint8)
                log_dict = {}
                log_dict["generated_videos"] = wandb.Video(frames, fps=args.fps)
                log_dict["gt_bbox_frames"] = wandb.Video(sample['bbox_img_np'], fps=args.fps)
                log_dict["gt_videos"] = wandb.Video(sample['gt_clip_np'], fps=args.fps)
                frame_bboxes = wandb_frames_with_bbox(frames, sample['objects_tensors'], (dataset.orig_W, dataset.orig_H))
                log_dict["frames_with_bboxes_{}".format(sample_i)] = frame_bboxes
                accelerator.trackers[0].log(log_dict)

                tmp = np.moveaxis(np.transpose(frames, (0, 2, 3, 1)), 0, 0)
                export_to_video(tmp, f"{plot_dir}/generated_ctrl_{sample_i}.mp4", fps=args.fps)
                with open(os.path.join(save_gt_path, f'sample_{sample_i}.pkl'), 'wb') as handle:
                    pickle.dump(sample['gt_labels'], handle, protocol=pickle.HIGHEST_PROTOCOL)
                if sample_i >= args.num_demo_samples:
                    break
                
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            with torch.autocast(
                str(accelerator.device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"
            ):
                from ctrlv.models import UNetSpatioTemporalConditionModel, ControlNetModel
                ctrlnet = ControlNetModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="controlnet")
                unet = UNetSpatioTemporalConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
                pipeline = StableVideoControlPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt",
                                                                      controlnet = ctrlnet,
                                                                      unet = unet,)
                # Run a final round of inference
                logger.info("Start inference...")
                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)

                run_inference_with_pipeline(pipeline, sample_generator)
        
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