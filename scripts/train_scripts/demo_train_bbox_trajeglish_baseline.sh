DATASET="..."  # [kitti, vkitti, bdd100k, nuscenes]
DATASET_PATH="..."
RUN_NAME="..."

python /home/mila/a/anthony.gosselin/dev/Ctrl-V_dev/src/ctrlv/bbox_generator_baseline/train.py \
    run_name=$RUN_NAME \
    initial_frames_condition_num=3 \
    use_state_embeddings=True \
    wandb_track=True \
    seed=0 \
    dataloader_workers=4 \
    train_batch_size=16 \
    val_batch_size=16 \
    map_embedding=True \  # NOTE: Can disable to not use initial image embeddings
    dataset=$DATASET \
    max_num_agents=15 \  # NOTE: Set to 30 for BDD100k dataset
    dataset_non_overlapping=False \
    pred_coords=False \
    always_predict_initial_agents=False \
    data_root=$DATASET_PATH \
    max_steps=20000