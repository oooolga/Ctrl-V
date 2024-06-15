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

import torch
torch.cuda.empty_cache()
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from ctrlv.utils import parse_args, get_dataloader, eval_samples_generator, eval_demo_samples_generator, wandb_frames_with_bbox
    from ctrlv.pipelines import VideoDiffusionPipeline

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
        sample_generator = eval_samples_generator(data_loader)
        if not args.demo_path is None:
            demo_pkls = []
            for file in os.listdir(args.demo_path):
                if file.endswith(".pkl"):
                    demo_pkls.append(os.path.join(args.demo_path, file))
            sample_generator = eval_demo_samples_generator(demo_pkls)


        def run_inference_with_pipeline(pipeline, sample_generator):
            
            from ctrlv.metrics import f_measure
            f_means_per_clip = []
            log_dict = defaultdict(list)
            for sample_i, sample in tqdm(enumerate(sample_generator)):
                frames = pipeline(sample['image_init'], 
                                height=dataset.train_H, width=dataset.train_W, 
                                bbox_images=sample['bbox_img'].unsqueeze(0),
                                decode_chunk_size=8, motion_bucket_id=127, fps=args.fps, 
                                num_inference_steps=args.num_inference_steps,
                                num_frames=args.clip_length,
                                min_guidance_scale=args.min_guidance_scale,
                                max_guidance_scale=args.max_guidance_scale,
                                noise_aug_strength=args.noise_aug_strength,
                                generator=generator, output_type='pt').frames[0]
                if args.demo_path is None:
                    f_means_per_frame = []
                    frames_cp = torch.where(frames<float(5/255), 0., frames)
                    for frame, bbox_frame in zip(frames_cp, sample['bbox_img_np']):
                        frame = to_pil_image(frame).convert('L')
                        bbox_frame = bbox_frame.transpose(1, 2, 0)
                        bbox_frame = to_pil_image(bbox_frame).convert('L')
                        f_means_per_frame.append(f_measure(np.asarray(frame), np.asarray(bbox_frame)))
                    
                    clip_f_mean = np.mean(f_means_per_frame)
                    f_means_per_clip.append(clip_f_mean)
                    print(clip_f_mean)
                else:
                    
                    frames = frames.detach().cpu().numpy()*255
                    frames = frames.astype(np.uint8)
                    log_dict["generated_videos"].append(wandb.Video(frames, fps=args.fps))
                    log_dict["gt_bbox_frames"].append(wandb.Video(sample['bbox_img_np'], fps=args.fps))
                    log_dict["gt_videos"].append(wandb.Video(sample['gt_clip_np'], fps=args.fps))
                    frame_bboxes = wandb_frames_with_bbox(frames, sample['objects_tensors'], (dataset.orig_W, dataset.orig_H))
                    log_dict["frames_with_bboxes_{}".format(sample_i)] = frame_bboxes

            return np.mean(f_means_per_clip) if args.demo_path is None else log_dict
                
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            with torch.autocast(
                str(accelerator.device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"
            ):
                from ctrlv.models import UNetSpatioTemporalConditionModel
                if not os.path.isfile(os.path.join(args.pretrained_model_name_or_path, "unet")):
                    # Get the most recent checkpoint
                    dirs = os.listdir(args.pretrained_model_name_or_path)
                    dirs = [d for d in dirs if d.startswith("checkpoint")]
                    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                    path = dirs[-1] if len(dirs) > 0 else None
                    args.pretrained_model_name_or_path = os.path.join(args.pretrained_model_name_or_path, path)
                unet = UNetSpatioTemporalConditionModel.from_pretrained(args.pretrained_model_name_or_path,
                                                                        subfolder="unet",
                                                                        low_cpu_mem_usage=True, num_frames=args.clip_length)
                pipeline = VideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt",
                                                                  unet = unet,)
                # Run a final round of inference
                logger.info("Start inference...")
                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)
                if args.demo_path is None:
                    f_mean = run_inference_with_pipeline(pipeline, sample_generator)
                    logger.info(f"Final f-measure: {f_mean}")
                else:
                    log_dict = run_inference_with_pipeline(pipeline, sample_generator)
                    
                    for tracker in accelerator.trackers:
                        if tracker.name == "wandb":
                            tracker.log(log_dict)
        
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