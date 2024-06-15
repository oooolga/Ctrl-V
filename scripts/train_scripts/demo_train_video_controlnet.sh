# nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9

timestamp=$(date +%y%m%d_%H%M%S)
DATASET="bdd100k" #"kitti/vkitti/bdd100k/..."
DATASET_PATH="..."
NAME="${DATASET}_ctrlv_${timestamp}"
OUT_DIR=".../${NAME}"
mkdir -p $OUT_DIR

PROJECT_NAME='ctrl_v'

SCRIPT_PATH=$0
SAVE_SCRIPT_PATH="${OUT_DIR}/train_scripts.sh"
cp $SCRIPT_PATH $SAVE_SCRIPT_PATH
echo "Saved script to ${SAVE_SCRIPT_PATH}"

CUDA_LAUNCH_BLOCKING=1 accelerate launch tools/train_video_controlnet.py \
    --run_name $NAME \
    --data_root $DATASET_PATH \
    --project_name $PROJECT_NAME \
    --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid-xt \
    --output_dir $OUT_DIR \
    --variant fp16 \
    --dataset_name $DATASET \
    --train_batch_size 1 \
    --learning_rate 1e-5 \
    --checkpoints_total_limit 2 \
    --checkpointing_steps 100 \
    --gradient_accumulation_steps 5 \
    --validation_steps 100 \
    --enable_gradient_checkpointing \
    --lr_scheduler constant \
    --report_to wandb \
    --seed 1234 \
    --mixed_precision fp16 \
    --clip_length 25 \
    --min_guidance_scale 1.0 \
    --max_guidance_scale 3.0 \
    --noise_aug_strength 0.01 \
    --bbox_dropout_prob 0.1 \
    --num_demo_samples 15 \
    --num_train_epochs 1 \
    --resume_from_checkpoint latest