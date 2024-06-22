def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument(
        "--project_name",
        type=str,
        default="ctrlv",
        help="Name of the project."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="eval",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--object_net_lr_factor",
        type=float,
        default=1.0,
        help="Factor to scale the learning rate of the object network.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="kitti",
        choices=['kitti', 'vkitti', 'mkitti', 'cityscapes', 'bdd100k', 'waymo', 'nuscenes', 'lyft', 'davis'],
        help=(
            "The name of the Dataset to train on."
        ),
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="The root directory of the dataset.",
    )
    
    # parser.add_argument(
    #     "--validation_prompt",
    #     type=str,
    #     default=None,
    #     help="A prompt that is used during validation to verify that the model is learning.",
    # )
    # parser.add_argument(
    #     "--num_validation_images",
    #     type=int,
    #     default=4,
    #     help="Number of images that should be generated during validation with `validation_prompt`.",
    # )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--disable_object_condition", action="store_true", help="Whether or not to disable object condition.")
    parser.add_argument("--encoder_hid_dim_type", type=str, default=None, help="The type of unet's input hidden.")
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="The name of the run log.",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="(Image only). A higher guidance scale value encourages the model to generate images closely linked to the text `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`."
    )
    parser.add_argument(
        "--guidance_rescale",
        type=float,
        default=0.0,
        help="(Image only). Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when using zero terminal SNR."
    )
    parser.add_argument(
        "--min_guidance_scale",
        type=float,
        default=1.0,
        help="(Video generation only). The minimum guidance scale. Used for the classifier free guidance with first frame."
    )
    parser.add_argument(
        "--max_guidance_scale",
        type=float,
        default=3.0,
        help="(Video generation only). The maximum guidance scale. Used for the classifier free guidance with last frame."
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0.1,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--clip_length",
        type=int,
        default=25,
        help="The number of frames in a clip.",
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--backprop_temporal_blocks_start_iter",
        type=int,
        default=-1,
        help="(Video generation only). The starting iteration of only backpropagating into temporal blocks (if -1, then always backpropagate the entire network).",
    )
    parser.add_argument(
        "--enable_lora",
        action="store_true",
        default=False,
        help="Enable LoRA.",
    )
    parser.add_argument(
        "--add_bbox_frame_conditioning",
        action="store_true",
        default=False,
        help="(Video generation only). Add bbox frame conditioning.",
    )
    parser.add_argument(
        "--bbox_dropout_prob",
        type=float,
        default=0.0,
        help="(Video generation only). Bbox dropout probability. Drops out the bbox conditionings.",
    )
    parser.add_argument(
        "--num_demo_samples",
        type=int,
        default=1,
        help="Number of samples to generate during demo.",
    )
    parser.add_argument(
        "--noise_aug_strength",
        type=float,
        default=0.02,
        help="(Video generation only). Strength of noise augmentation.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        help="Number of inference denoising steps.",
    )
    parser.add_argument(
        "--conditioning_scale",
        type=float,
        default=1.0,
        help="(Controlnet only). The scale of conditioning."
    )
    parser.add_argument(
        "--train_H",
        type=int,
        default=None,
        help="For training, the height of the image to use. If None, the default height is 320 for video diffusion and 512 for image diffusion."
    )
    parser.add_argument(
        "--train_W",
        type=int,
        default=None,
        help="For training, the width of the image to use. If None, the default width is 512."
    )
    parser.add_argument(
        "--eval_H",
        type=int,
        default=None,
        help="For evaluation, the height of the image to use. If None, the default height is 320 for video diffusion and 512 for image diffusion."
    )
    parser.add_argument(
        "--generate_bbox",
        action="store_true",
        default=False,
        help="(Controlnet only). Whether to generate bbox."
    )
    parser.add_argument(
        "--predict_bbox",
        action="store_true",
        default=False,
        help="(Video diffusion only). Whether to predict bbox."
    )
    parser.add_argument(
        "--evaluate_only",
        action="store_true",
        default=False,
        help="Whether to only evaluate the model."
    )
    parser.add_argument(
        "--use_segmentation",
        action="store_true",
        default=False,
        help="(bdd100k only). Whether to use segmentation instead of bbox."
    )
    parser.add_argument(
        "--demo_path",
        default=None,
        type=str,
        help="Path where the demo samples are saved."
    )
    parser.add_argument(
        "--pretrained_bbox_model",
        default=None,
        type=str,
        help="Path to the pretrained bbox model."
    )
    parser.add_argument(
        "--if_last_frame_trajectory",
        action="store_true",
        default=False,
        help="Whether to use the last frame as the trajectory."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="FPS of the video."
    )
    parser.add_argument(
        "--num_cond_bbox_frames",
        type=int,
        default=3,
        help="Number of conditioning bbox frames."
    )
    args = parser.parse_args()
    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision
    
    if args.enable_lora:
        args.backprop_temporal_blocks_start_iter = -1
    
    if args.evaluate_only:
        assert args.resume_from_checkpoint is not None, "Must provide a checkpoint to evaluate the model."
    
    if args.use_segmentation:
        assert args.dataset_name == "bdd100k" or args.dataset_name == "davis", "Segmentation is only supported for bdd100k dataset."
    
    if args.fps is None:
        if args.dataset_name == "bdd100k":
            args.fps = 5
        else:
            args.fps = 7
    return args