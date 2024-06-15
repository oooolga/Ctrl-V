DATASET='...'
OUT_DIR="..."
DATASET_PATH="..."

PROJECT_NAME='ctrl_v'

accelerate launch --config_file config/a100l.yaml tools/eval_video_controlnet.py \
    --run_name $DATASET-box2video-tf-final-eval \
    --data_root $DATASET_PATH \
    --project_name $PROJECT_NAME \
    --pretrained_model_name_or_path $OUT_DIR \
    --output_dir $OUT_DIR \
    --variant fp16 \
    --dataset_name $DATASET \
    --report_to wandb \
    --seed 123 \
    --mixed_precision fp16 \
    --clip_length 25 \
    --min_guidance_scale 1.0 \
    --max_guidance_scale 3.0 \
    --noise_aug_strength 0.01 \
    --bbox_dropout_prob 0.1 \
    --num_demo_samples 200 \
    --num_inference_steps 30 \
    --conditioning_scale 1.0 \
    --train_batch_size 1 \
    --resume_from_checkpoint latest