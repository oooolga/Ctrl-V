OUT_DIR="..."
DATASET_PATH="..."
DATASET="..."
PROJECT_NAME='ctrl-v'

accelerate launch --config_file config/rtx8000.yaml tools/eval_video_generation.py \
    --run_name $DATASET-baseline-final-eval \
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
    --train_batch_size 1 \
    --bbox_dropout_prob 0.1 \
    --num_demo_samples 200 \
    --resume_from_checkpoint latest \
    --num_inference_steps 50