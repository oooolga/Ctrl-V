MODEL_PATH="..."  # Where the checkpoint to evaluate is saved (.ckpt file)
DATASET="..." # [kitti, vkitti, bdd100k, nuscenes]

python src/ctrlv/bbox_prediction/eval.py \
    initial_frames_condition_num=3 \
    seed=0 \
    val_batch_size=1 \
    map_embedding=True \
    dataset=$DATASET \
    dataset_non_overlapping=True \
    model_path=$MODEL_PATH \
    num_eval_samples=200