# nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9

timestamp=$(date +%y%m%d_%H%M%S)
DATASET='vkitti'
NAME="${DATASET}_box_predict_${timestamp}"
OUT_DIR="/network/scratch/x/xuolga/Results/sd3d/${NAME}"
mkdir -p $OUT_DIR

PROJECT_NAME='sd3d-video'

SCRIPT_PATH=$0
SAVE_SCRIPT_PATH="${OUT_DIR}/train_scripts.sh"
cp $SCRIPT_PATH $SAVE_SCRIPT_PATH
echo "Saved script to ${SAVE_SCRIPT_PATH}"

CUDA_LAUNCH_BLOCKING=1 accelerate launch tools/train_video_diffusion.py \
    --run_name $NAME \
    --data_root $SCRATCH/Datasets \
    --project_name $PROJECT_NAME \
    --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid-xt \
    --output_dir $OUT_DIR \
    --variant fp16 \
    --dataset_name $DATASET \
    --train_batch_size 1 \
    --learning_rate 5e-6 \
    --checkpoints_total_limit 2 \
    --checkpointing_steps 200 \
    --gradient_accumulation_steps 5 \
    --validation_steps 100 \
    --enable_gradient_checkpointing \
    --lr_scheduler constant \
    --report_to wandb \
    --seed 1234 \
    --mixed_precision fp16 \
    --clip_length 25 \
    --min_guidance_scale 3 \
    --max_guidance_scale 7 \
    --noise_aug_strength 0.01 \
    --bbox_dropout_prob 0.1 \
    --conditioning_dropout_prob 0.0 \
    --num_demo_samples 10 \
    --backprop_temporal_blocks_start_iter -1 \
    --num_train_epochs 5 \
    --predict_bbox \
    --num_inference_steps 30 \
    --resume_from_checkpoint latest \
    --num_cond_bbox_frames 1
    # --if_last_frame_trajectory
    # --use_segmentation