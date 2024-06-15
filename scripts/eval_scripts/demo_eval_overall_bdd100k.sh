DATASET='bdd100k'
OUT_DIR="/network/scratch/x/xuolga/Results/sd3d/bdd100k_ctrlv_240511_200727"
BBOX_MODEL_DIR="/network/scratch/x/xuolga/Results/sd3d/bdd100k_segmentation_predict_240515_102502_backup"

PROJECT_NAME="ctrl_v"

accelerate launch --config_file config/a100l.yaml tools/eval_overall.py \
    --data_root /network/scratch/x/xuolga/Datasets \
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
    --max_guidance_scale 5.0 \
    --noise_aug_strength 0.01 \
    --train_batch_size 1 \
    --num_demo_samples 300 \
    --resume_from_checkpoint latest \
    --num_inference_steps 50 \
    --pretrained_bbox_model $BBOX_MODEL_DIR \
    --num_cond_bbox_frames 1