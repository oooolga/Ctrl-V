DATASET='bdd100k'
OUT_DIR='/network/scratch/x/xuolga/Results/sd3d/bdd100k_segmentation_predict_240515_050141'

PROJECT_NAME='sd3d-video'

accelerate launch --config_file config/rtx8000.yaml tools/eval_video_bbox_prediction.py \
    --data_root $SCRATCH/Datasets \
    --project_name $PROJECT_NAME \
    --pretrained_model_name_or_path $OUT_DIR \
    --output_dir $OUT_DIR \
    --variant fp16 \
    --dataset_name $DATASET \
    --report_to wandb \
    --seed 4321 \
    --mixed_precision fp16 \
    --clip_length 25 \
    --min_guidance_scale 1.0 \
    --max_guidance_scale 3.0 \
    --noise_aug_strength 0.01 \
    --train_batch_size 1 \
    --bbox_dropout_prob 0.1 \
    --num_demo_samples 25 \
    --resume_from_checkpoint latest \
    --use_segmentation