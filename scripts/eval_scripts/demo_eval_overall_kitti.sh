DATASET='kitti'
OUT_DIR="/network/scratch/x/xuolga/Results/sd3d/kitti_ctrlv_240510_141159"
#BBOX_MODEL_DIR="/network/scratch/x/xuolga/Results/sd3d/kitti_box_predict_240601_152554"
BBOX_MODEL_DIR="/home/mila/x/xuolga/scratch/Results/sd3d/kitti_box_predict_240609_142815"

PROJECT_NAME="ctrl_v_final_eval"

accelerate launch --config_file config/a100l.yaml tools/eval_overall.py \
    --data_root /network/scratch/x/xuolga/Datasets \
    --project_name $PROJECT_NAME \
    --pretrained_model_name_or_path $OUT_DIR \
    --output_dir $OUT_DIR \
    --variant fp16 \
    --dataset_name $DATASET \
    --report_to wandb \
    --seed 1234 \
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
    --num_cond_bbox_frames 1