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

from tqdm.auto import tqdm

import torch
torch.cuda.empty_cache()
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from packaging import version
from diffusers.utils.import_utils import is_xformers_available
from torchvision.transforms.functional import to_pil_image

from diffusers import EulerDiscreteScheduler
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
    from ctrlv.models import UNetSpatioTemporalConditionModel, ControlNetModel
    from ctrlv.pipelines import StableVideoControlPipeline

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
        mixed_precision=args.mixed_precision,
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
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        # Load scheduler, tokenizer and models.
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant="fp16",
        )

        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", variant="fp16",
            low_cpu_mem_usage=True, num_frames=args.clip_length
        )

        feature_extractor = CLIPImageProcessor.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision,
        )

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision, variant="fp16"
        )

        # freeze parameters of models to save more memory
        vae.requires_grad_(False)
        image_encoder.requires_grad_(False)
        unet.requires_grad_(False)
        # Load the model
        ctrlnet = ControlNetModel.from_unet(unet)

        # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = ctrlnet_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Setup devices
        num_devices = torch.cuda.device_count()
        available_devices = [torch.device(i) for i in range(num_devices)]
        accelerator_device = accelerator.device

        if num_devices > 1:
            other_devices = [device for device in available_devices if device != accelerator_device]
        vae_device = other_devices[0] if num_devices > 1 else accelerator_device
        
        # Move unet, vae and text_encoder to device and cast to weight_dtype
        vae.to(vae_device, dtype=weight_dtype)
        image_encoder.to(vae_device, dtype=weight_dtype)
        unet.to(accelerator_device, dtype=weight_dtype)

        # `accelerate` 0.16.0 will have better support for customized saving
        if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                if accelerator.is_main_process:

                    for i, model in enumerate(models):
                        if isinstance(model, UNetSpatioTemporalConditionModel):
                            model.save_pretrained(os.path.join(output_dir, "unet"), safe_serialization=False)
                        elif isinstance(model, ControlNetModel):
                            model.save_pretrained(os.path.join(output_dir, "control_net"), safe_serialization=False)
                        else:
                            raise Exception("Only UNetSpatioTemporalConditionModel and ControlNetModel are supported for saving.")

                        # make sure to pop weight so that corresponding model is not saved again
                        weights.pop()
            
            def load_model_hook(models, input_dir):

                for _ in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()
                    if isinstance(model, UNetSpatioTemporalConditionModel):
                        load_model = UNetSpatioTemporalConditionModel.from_pretrained(input_dir, subfolder="unet")
                    elif isinstance(model, ControlNetModel):
                        load_model = ControlNetModel.from_pretrained(input_dir, subfolder="control_net")
                    else:
                        raise Exception("Only UNetSpatioTemporalConditionModel and ControlNetModel are supported for loading.")
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model

            accelerator.register_save_state_pre_hook(save_model_hook)
            accelerator.register_load_state_pre_hook(load_model_hook)


        if args.enable_gradient_checkpointing:
            unet.enable_gradient_checkpointing()

        if args.scale_lr:
            args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps *
                args.per_gpu_batch_size * accelerator.num_processes
            )

        ctrlnet.requires_grad_(True)

        optimizer = torch.optim.AdamW(
            ctrlnet.parameters(), 
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        train_dataset, train_loader = get_dataloader(args.data_root, args.dataset_name, if_train=True, clip_length=args.clip_length,
                                                    batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers, 
                                                    data_type='clip', use_default_collate=True, tokenizer=None, shuffle=True,
                                                    if_return_bbox_im=True, train_H=args.train_H, train_W=args.train_W,
                                                    use_segmentation=args.use_segmentation)
        # _, test_loader = get_dataloader(args.dataset_name, if_train=True, 
        #                                 batch_size=1, num_workers=args.dataloader_num_workers, 
        #                                 data_type='clip', use_default_collate=True, tokenizer=None, shuffle=True)
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
        unet, ctrlnet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
            unet, ctrlnet, optimizer, train_loader, lr_scheduler
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
        
        def get_sigmas(timesteps, n_dim=5, dtype=torch.float32):
            sigmas = noise_scheduler.sigmas.to(device=accelerator_device, dtype=dtype)
            schedule_timesteps = noise_scheduler.timesteps.to(accelerator_device)
            timesteps = timesteps.to(accelerator_device)

            step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

            sigma = sigmas[step_indices].flatten()
            while len(sigma.shape) < n_dim:
                sigma = sigma.unsqueeze(-1)
            return sigma
        
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

        demo_samples = get_n_training_samples(train_loader, args.num_demo_samples)

        generator = torch.Generator(device=accelerator_device).manual_seed(args.seed) if args.seed else None

        def run_inference_with_pipeline(pipeline, demo_samples, log_dict):
            for sample_i, sample in enumerate(demo_samples):
                frames = pipeline(sample['image_init'] if not args.generate_bbox else sample['bbox_init'], 
                                cond_images=sample['bbox_img'].unsqueeze(0) if not args.generate_bbox else sample['gt_clip'].unsqueeze(0),
                                height=train_dataset.train_H, width=train_dataset.train_W, 
                                decode_chunk_size=8, motion_bucket_id=127, fps=args.fps, 
                                num_inference_steps=args.num_inference_steps,
                                num_frames=args.clip_length,
                                control_condition_scale=args.conditioning_scale,
                                min_guidance_scale=args.min_guidance_scale,
                                max_guidance_scale=args.max_guidance_scale,
                                noise_aug_strength=args.noise_aug_strength,
                                generator=generator, output_type='pt').frames[0]
                #frames = F.interpolate(frames, (train_dataset.orig_H, train_dataset.orig_W)).detach().cpu().numpy()*255
                frames = frames.detach().cpu().numpy()*255
                frames = frames.astype(np.uint8)
                log_dict["generated_videos"].append(wandb.Video(frames, fps=args.fps))
                log_dict["gt_bbox_frames"].append(wandb.Video(sample['bbox_img_np'], fps=args.fps))
                log_dict["gt_videos"].append(wandb.Video(sample['gt_clip_np'], fps=args.fps))
                frame_bboxes = wandb_frames_with_bbox(frames, sample['objects_tensors'], (train_dataset.orig_W, train_dataset.orig_H))
                log_dict["frames_with_bboxes_{}".format(sample_i)] = frame_bboxes
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
                                pipeline = StableVideoControlPipeline.from_pretrained(args.pretrained_model_name_or_path,
                                                                            unet=unwrap_model(unet), 
                                                                            image_encoder=unwrap_model(image_encoder),
                                                                            vae=unwrap_model(vae),
                                                                            controlnet=unwrap_model(ctrlnet),
                                                                            feature_extractor=feature_extractor,
                                                                            revision=args.revision, 
                                                                            variant=args.variant, 
                                                                            torch_dtype=weight_dtype,
                                                                            )
                                ctrlnet.eval()
                                pipeline = pipeline.to(accelerator_device)
                                pipeline.set_progress_bar_config(disable=True)
                                log_dict = run_inference_with_pipeline(pipeline, demo_samples, log_dict)

                            for tracker in accelerator.trackers:
                                if tracker.name == "wandb":
                                    tracker.log(log_dict)
                            del pipeline, log_dict
                            torch.cuda.empty_cache()
                
                ctrlnet.train()
                with accelerator.accumulate(ctrlnet):
                    # Forward pass
                    batch_size, video_length = batch['clips'].shape[0], batch['clips'].shape[1]
                    initial_images = batch['clips'][:,0,:,:,:] if not args.generate_bbox else batch['bbox_images'][:,0,:,:,:] # only use the first frame
                    # check device
                    if vae.device != vae_device:
                        vae.to(vae_device)
                        image_encoder.to(vae_device)
                        initial_images.to(vae_device)
                    
                    # Encode input image
                    encoder_hidden_states = encode_video_image(initial_images, feature_extractor, weight_dtype, image_encoder).unsqueeze(1)
                    encoder_hidden_states = encoder_hidden_states.to(dtype=ctrlnet_dtype).to(accelerator_device)

                    # Encode input image using VAE
                    conditional_latents = vae.encode(initial_images.to(vae_device).to(weight_dtype)).latent_dist.sample()
                    conditional_latents = conditional_latents.to(accelerator_device).to(ctrlnet_dtype)

                    # Encode bbox image using VAE
                    # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
                    bbox_frames = rearrange(batch['bbox_images'] if not args.generate_bbox else batch['clips'], 'b f c h w -> (b f) c h w').to(vae.device).to(weight_dtype)
                    bbox_em = vae.encode(bbox_frames).latent_dist.sample()
                    bbox_em = rearrange(bbox_em, '(b f) c h w -> b f c h w', f=video_length).to(accelerator_device).to(ctrlnet_dtype)

                    # Encode clip frames using VAE
                    # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
                    frames = rearrange(batch['clips'] if not args.generate_bbox else batch['bbox_images'], 'b f c h w -> (b f) c h w').to(vae.device).to(weight_dtype)
                    latents = vae.encode(frames).latent_dist.sample()
                    latents = rearrange(latents, '(b f) c h w -> b f c h w', f=video_length).to(accelerator_device).to(ctrlnet_dtype)
                    target_latents = latents = latents * vae.config.scaling_factor

                    del batch, frames
                    noise = torch.randn_like(latents)
                    
                    indices = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=noise_scheduler.timesteps.device).long()
                    timesteps = noise_scheduler.timesteps[indices].to(accelerator_device)

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Scale the noisy latents for the UNet
                    sigmas = get_sigmas(timesteps, len(noisy_latents.shape), noisy_latents.dtype)
                    # inp_noisy_latents = noise_scheduler.scale_model_input(noisy_latents, timesteps)
                    inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)

                    added_time_ids = get_add_time_ids(
                        fps=args.fps-1,
                        motion_bucket_id=127,
                        noise_aug_strength=args.noise_aug_strength,
                        dtype=weight_dtype,
                        batch_size=batch_size,
                        unet=unet
                    ).to(accelerator_device)
                    
                    # Conditioning dropout to support classifier-free guidance during inference. For more details
                    # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                    # Addapted from https://github.com/huggingface/diffusers/blob/0d2d424fbef933e4b81bea20a660ee6fc8b75ab0/docs/source/en/training/instructpix2pix.md
                    if args.conditioning_dropout_prob is not None:
                        random_p = torch.rand(
                            batch_size, device=accelerator_device, generator=generator)
                        # Sample masks for the edit prompts.
                        prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                        prompt_mask = prompt_mask.reshape(batch_size, 1, 1)
                        # Final text conditioning.
                        null_conditioning = torch.zeros_like(encoder_hidden_states)
                        encoder_hidden_states = torch.where(
                            prompt_mask, null_conditioning, encoder_hidden_states)
                        # Sample masks for the original images.
                        image_mask_dtype = conditional_latents.dtype
                        image_mask = 1 - (
                            (random_p >= args.conditioning_dropout_prob).to(
                                image_mask_dtype)
                            * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                        )
                        image_mask = image_mask.reshape(batch_size, 1, 1, 1)
                        # Final image conditioning.
                        conditional_latents = image_mask * conditional_latents

                    # Concatenate the `original_image_embeds` with the `noisy_latents`.
                    conditional_latents = unet.encode_bbox_frame(conditional_latents, None)
                    
                    concatenated_noisy_latents = torch.cat([inp_noisy_latents, conditional_latents], dim=2)
                    added_time_ids = added_time_ids.to(dtype=ctrlnet_dtype)
                    
                    down_block_additional_residuals, mid_block_additional_residuals = ctrlnet(
                        concatenated_noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        added_time_ids=added_time_ids,
                        control_cond=bbox_em,
                        conditioning_scale=args.conditioning_scale,
                        return_dict=False,
                    )
                    model_pred = unet(sample=concatenated_noisy_latents,
                                    timestep=timesteps,
                                    encoder_hidden_states=encoder_hidden_states, 
                                    added_time_ids=added_time_ids,
                                    down_block_additional_residuals=down_block_additional_residuals,
                                    mid_block_additional_residuals=mid_block_additional_residuals,).sample

                    # Denoise the latents
                    c_out = -sigmas / ((sigmas**2 + 1)**0.5)
                    c_skip = 1 / (sigmas**2 + 1)
                    denoised_latents = model_pred * c_out + c_skip * noisy_latents
                    weighting = (1 + sigmas ** 2) * (sigmas**-2.0)

                    # # MSE loss
                    loss = torch.mean(
                        (weighting.float() * (denoised_latents.float() - target_latents.float()) ** 2).reshape(target_latents.shape[0], -1),
                        dim=1,
                    )
                    loss = loss.mean()

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                del loss, latents, concatenated_noisy_latents, model_pred, weighting, inp_noisy_latents, noisy_latents, timesteps, indices, sigmas, added_time_ids

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    log_plot = {
                                "train_loss": train_loss, 
                                "lr": lr_scheduler.get_last_lr()[0],
                            }
                    if args.add_bbox_frame_conditioning:
                        log_plot["|attn_rz_weight|"] = unet.get_attention_rz_weight()
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
        if accelerator.is_main_process:
            with torch.autocast(
                str(accelerator_device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"
            ):
                ctrlnet = unwrap_model(ctrlnet)
                unet = unwrap_model(unet)
                pipeline = StableVideoControlPipeline.from_pretrained(args.pretrained_model_name_or_path,
                                                                    unet=unet, 
                                                                    image_encoder=unwrap_model(image_encoder),
                                                                    vae=unwrap_model(vae),
                                                                    controlnet=ctrlnet,
                                                                    feature_extractor=feature_extractor,
                                                                    revision=args.revision, 
                                                                    variant=args.variant, 
                                                                    torch_dtype=weight_dtype,
                                                                    )
                pipeline.save_pretrained(args.output_dir)

                # Run a final round of inference
                logger.info("Running inference before terminating...")
                pipeline = pipeline.to(accelerator_device)
                pipeline.torch_dtype = weight_dtype
                pipeline.set_progress_bar_config(disable=True)

                log_dict = defaultdict(list)
                log_dict = run_inference_with_pipeline(pipeline, demo_samples, log_dict)
                for tracker in accelerator.trackers:
                    if tracker.name == "wandb":
                        tracker.log(log_dict)
        
        logging.info("Finished training.")
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