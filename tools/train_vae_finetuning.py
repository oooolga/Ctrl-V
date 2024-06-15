from accelerate.utils import write_basic_config
write_basic_config()
import warnings

import logging
import os
import math
import shutil
from pathlib import Path
import numpy as np
from einops import rearrange
import accelerate
from collections import defaultdict
import itertools

from tqdm.auto import tqdm
from peft import LoraConfig

import torch
torch.cuda.empty_cache()
import torch.nn.functional as F
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from diffusers.utils.import_utils import is_xformers_available
from torchvision.transforms.functional import to_pil_image

from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.training_utils import EMAModel

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from ctrlv.utils import parse_args, get_dataloader, encode_video_image, get_add_time_ids, get_fourier_embeds_from_boundingbox, get_n_training_samples, wandb_frames_with_bbox

if not is_wandb_available():
    warnings.warn("Make sure to install wandb if you want to use it for logging during training.")
else: 
    import wandb
logger = get_logger(__name__, log_level="INFO")

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

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

        # Handle the repository creation
        if accelerator.is_main_process:
            if args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)
                plot_dir = os.path.join(args.output_dir, "plots")
                os.makedirs(plot_dir, exist_ok=True)

        # Load scheduler, tokenizer and models.
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant="fp16",
        )

        # freeze parameters of models to save more memory
        vae.requires_grad_(False)
        vae_trainable_params = []
        for name, param in vae.named_parameters():
            if 'decoder' in name:
                param.requires_grad = True
                vae_trainable_params.append(param)
        param_size = 0
        for name, param in vae.named_parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for name, buffer in vae.named_buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size = param_size + buffer_size
        logger.info(f"Model size: {size / 1024 / 1024:.2f} MB")

        # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32

        # Setup devices
        accelerator_device = accelerator.device
        # Move unet, vae and text_encoder to device and cast to weight_dtype
        vae.to(accelerator_device, dtype=weight_dtype)

        # `accelerate` 0.16.0 will have better support for customized saving
        if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                if accelerator.is_main_process:

                    for i, model in enumerate(models):
                        if isinstance(model, AutoencoderKLTemporalDecoder):
                            model.save_pretrained(os.path.join(output_dir, "vae"), safe_serialization=False)
                        else:
                            raise Exception("Only AutoencoderKLTemporalDecoder issupported for saving.")
                        # make sure to pop weight so that corresponding model is not saved again
                        weights.pop()
            
            def load_model_hook(models, input_dir):

                for _ in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()
                    if isinstance(model, AutoencoderKLTemporalDecoder):
                        load_model = AutoencoderKLTemporalDecoder.from_pretrained(input_dir, subfolder="vae")
                    else:
                        raise Exception("Only AutoencoderKLTemporalDecoder is supported for loading.")
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model

            accelerator.register_save_state_pre_hook(save_model_hook)
            accelerator.register_load_state_pre_hook(load_model_hook)

        if args.enable_gradient_checkpointing:
            vae.enable_gradient_checkpointing()

        if args.scale_lr:
            args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps *
                args.per_gpu_batch_size * accelerator.num_processes
            )

        params_to_optimize = (
            itertools.chain(vae_trainable_params)
        )
        optimizer = torch.optim.AdamW(
            params_to_optimize, 
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        train_dataset, train_loader = get_dataloader(args.data_root, args.dataset_name, if_train=True,
                                                    batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers, 
                                                    data_type='image', use_default_collate=True, tokenizer=None, shuffle=True,
                                                    if_return_bbox_im=True, train_H=args.train_H, train_W=args.train_W,
                                                    use_segmentation=args.use_segmentation)
        test_dataset, test_loader = get_dataloader(args.data_root, args.dataset_name, if_train=False,
                                                   batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers, 
                                                   data_type='image', use_default_collate=True, tokenizer=None, shuffle=False,
                                                   if_return_bbox_im=True, train_H=args.train_H, train_W=args.train_W,
                                                   use_segmentation=args.use_segmentation)
        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
        )
        
        # Prepare everything with our `accelerator`.
        vae, optimizer, train_loader, lr_scheduler = accelerator.prepare(
            vae, optimizer, train_loader, lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            accelerator.init_trackers(args.project_name, config=vars(args), init_kwargs={"wandb": {"dir": args.output_dir, "name": args.run_name}})
        
        # Train!
        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        initial_global_step = global_step = 0
        first_epoch = 0

        # # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint != "latest":
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None
            
            if path is None:
                accelerator.print(
                    f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                args.resume_from_checkpoint = None
                initial_global_step = 0
            else:
                accelerator.print(f"Resuming from checkpoint {path}")
                accelerator.load_state(os.path.join(args.output_dir, path))
                initial_global_step = global_step = int(path.split("-")[1])

                first_epoch = global_step // num_update_steps_per_epoch
        else:
            initial_global_step = 0

        progress_bar = tqdm(
            range(0, args.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )

        def unwrap_model(model):
            model = accelerator.unwrap_model(model)
            model = model._orig_mod if is_compiled_module(model) else model
            return model

        generator = torch.Generator(device=accelerator_device).manual_seed(args.seed) if args.seed else None

        def run_inference(vae, test_loader, dataset, log_dict, num_samples=5, generator=None):
            for _, batch in enumerate(test_loader):
                if args.predict_bbox:
                    input_images = batch["bbox_images"].to(vae.device)
                else:
                    input_images = batch["pixel_values"].to(vae.device)
                pred_images = vae(input_images.to(vae.dtype), sample_posterior=True).sample
                gt_images = dataset.revert_transform_no_resize(input_images)
                gt_images = gt_images.detach().cpu().numpy()*255
                gt_images = gt_images.astype(np.uint8)

                pred_images = pred_images.clamp(-1, 1)
                frames = dataset.revert_transform_no_resize(pred_images)
                frames = frames.detach().cpu().numpy()*255
                frames = frames.astype(np.uint8)
                log_dict['reconstructed_samples'] += wandb_frames_with_bbox(frames)
                log_dict['gt_samples'] += wandb_frames_with_bbox(gt_images)
                num_samples -= len(input_images)
                if num_samples <= 0:
                    break
            return log_dict
        
        for _ in range(first_epoch, args.num_train_epochs):
            
            train_loss = 0.0
            for _, batch in enumerate(train_loader):

                if accelerator.sync_gradients:
                    if accelerator.is_main_process:
                        
                        if global_step % args.validation_steps == 0:
                            logger.info("Running validation... ")

                            log_dict = defaultdict(list)
                            with torch.autocast(
                                    str(accelerator_device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"
                                ):
                                vae.eval()
                                log_dict = run_inference(unwrap_model(vae), test_loader, test_loader.dataset, log_dict, 
                                                         num_samples=args.num_demo_samples,
                                                         generator=generator)

                            for tracker in accelerator.trackers:
                                if tracker.name == "wandb":
                                    tracker.log(log_dict)
                            del log_dict
                            torch.cuda.empty_cache()
                
                vae.train()
                with accelerator.accumulate(vae):
                    # Forward pass
                    if args.predict_bbox:
                        input_images = batch["bbox_images"]
                    else:
                        input_images = batch["pixel_values"]
                    pred_images = vae(input_images, sample_posterior=True, generator=generator).sample
                    pred_images = pred_images.clamp(-1, 1)
                    
                    # # MSE loss
                    loss = F.mse_loss(pred_images.float(), input_images.float(), reduction="mean")

                    # Backpropagate
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    log_plot = {
                                "train_loss": loss.detach().item(), 
                                "lr": lr_scheduler.get_last_lr()[0],
                            }
                    accelerator.log(log_plot, step=global_step)
                    train_loss = 0.0

                    if global_step % args.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                if global_step >= args.max_train_steps:
                    break
            
        accelerator.wait_for_everyone()
        # if accelerator.is_main_process:
        #     with torch.autocast(
        #         str(accelerator_device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"
        #     ):
        #         pipeline = StableVideoControlPipeline.from_pretrained(args.pretrained_model_name_or_path,
        #                                                             unet=unet, 
        #                                                             image_encoder=unwrap_model(image_encoder),
        #                                                             vae=unwrap_model(vae),
        #                                                             controlnet=ctrlnet,
        #                                                             feature_extractor=feature_extractor,
        #                                                             revision=args.revision, 
        #                                                             variant=args.variant, 
        #                                                             torch_dtype=weight_dtype,
        #                                                             )
        #         pipeline.save_pretrained(args.output_dir)

        #         # Run a final round of inference
        #         logger.info("Running inference before terminating...")
        #         pipeline = pipeline.to(accelerator_device)
        #         pipeline.torch_dtype = weight_dtype
        #         pipeline.set_progress_bar_config(disable=True)

        #         log_dict = defaultdict(list)
        #         log_dict = run_inference_with_pipeline(pipeline, demo_samples, log_dict)
        #         for tracker in accelerator.trackers:
        #             if tracker.name == "wandb":
        #                 tracker.log(log_dict)
        
        # logging.info("Finished training.")
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