import os 
import hydra
from ctrlv.bbox_prediction.models.bbox_predictor_lm import BboxPredictorLM
from ctrlv.bbox_prediction.datamodules.datamodule import DataModule

import torch
torch.set_float32_matmul_precision('medium')
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger


@hydra.main(version_base=None, config_path="/home/mila/a/anthony.gosselin/dev/Ctrl-V_dev/src/ctrlv/bbox_prediction/cfgs/", config_name="config")
def main(cfg):
    pl.seed_everything(cfg.seed, workers=True)

    # Create model and datamodule
    model = BboxPredictorLM(cfg)
    datamodule = DataModule(cfg)

    # Create save directory for wandb stuff
    # save_dir = f'/home/wandb/{cfg.run_name}'
    save_dir = f'/home/mila/a/anthony.gosselin/wandb/diffuser/{cfg.run_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Used to save model periodically
    model_checkpoint = ModelCheckpoint(monitor='val_loss', save_top_k=cfg.save_top_k_checkpoints or 7, save_last=True, mode='min', dirpath=save_dir)

    # Used to log LR during training for debugging
    lr_monitor = LearningRateMonitor(logging_interval='step')

    model_summary = ModelSummary(max_depth=-1)

    logger = None
    if cfg.wandb_track:
        logger = WandbLogger(
            project='bbox_pred',
            name=cfg.run_name,
            # entity='', # TODO: Change for collaborative group name
            log_model=False,
            save_dir=save_dir
        )
    
    # Start training from checkpoint? (if cfg.run_name has already been used)
    ckpt_path = None
    if cfg.run_name != "default":
        files_in_save_dir = os.listdir(save_dir)
        for file in files_in_save_dir:
            if file.endswith('.ckpt') and 'last' in file:
                ckpt_path = os.path.join(save_dir, file)
                print("RESUMING TRAINING FROM CHECKPOINT: ", ckpt_path)
    
    trainer = pl.Trainer(accelerator=cfg.accelerator,
                         devices=cfg.num_devices,
                         callbacks=[model_summary, model_checkpoint, lr_monitor],
                         max_steps=cfg.max_steps,
                         max_epochs=50,
                         check_val_every_n_epoch=cfg.val_freq,
                         precision=cfg.precision,
                         limit_train_batches=cfg.train_data_fraction, 
                         limit_val_batches=cfg.val_data_fraction, 
                         gradient_clip_val=cfg.gradient_clip_val,
                         detect_anomaly=False,
                        #  profiler="simple", 
                         log_every_n_steps=cfg.log_every_n_steps,
                         logger=logger
                        )
    
    
    # Start training (training loop handled by pytorch_lightning)
    trainer.fit(model, datamodule, ckpt_path=ckpt_path)

if __name__ == '__main__':
    main()


