DATASET='kitti'
OUT_DIR="..."
BOX2VIDEO_DIR="..."
BBOX_MODEL_DIR="..."
DATASET_PATH="..."

mkdir -p $OUT_DIR
PROJECT_NAME="ctrl_v"

accelerate launch --config_file config/a100l.yaml tools/eval_overall.py \
    --data_root $DATASET_PATH \
    --project_name $PROJECT_NAME \
    --pretrained_model_name_or_path $BOX2VIDEO_DIR \
    --output_dir $OUT_DIR \
    --variant fp16 \
    --dataset_name $DATASET \
    --report_to wandb \
    --seed 123 \
    --mixed_precision fp16 \
    --clip_length 25 \
    --min_guidance_scale 1.0 \
    --max_guidance_scale 5.0 \
    --noise_aug_strength 0.01 \
    --train_batch_size 1 \
    --num_demo_samples 200 \
    --resume_from_checkpoint latest \
    --num_inference_steps 50 \
    --pretrained_bbox_model $BBOX_MODEL_DIR \
    --num_cond_bbox_frames 3  # NOTE: (3 for 3-to-1 variant, 1 for 1-to-1 variant)