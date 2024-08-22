import pytorch_lightning as pl 

from sd3d.utils.util import get_dataloader


class DataModule(pl.LightningDataModule):

    def __init__(self, cfg):
        super(DataModule, self).__init__()
        self.cfg = cfg
        self.dataloader_train = None
        self.dataloader_val = None
        self.dataset_train = None
        self.dataset_val = None

        self.disable_images = not self.cfg.map_embedding
        if self.cfg.disable_image_load:
            self.disable_images = True
    
    
    def setup(self, stage):
        pass # Data split is done when fetching dataloaders

    def train_dataloader(self):
        dataset_train, dataloader_train = get_dataloader(self.cfg.data_root, self.cfg.dataset, if_train=True, batch_size=self.cfg.train_batch_size, num_workers=self.cfg.dataloader_workers, 
                            data_type='clip', clip_length=self.cfg.num_timesteps, use_default_collate=True, tokenizer=None, shuffle=False, if_return_bbox_im=self.cfg.load_bbox_image, non_overlapping_clips=self.cfg.dataset_non_overlapping)
        self.dataset_train = dataset_train
        self.dataloader_train = dataloader_train

        if self.disable_images:
            dataset_train.disable_get_image()

        return dataloader_train


    def val_dataloader(self):
        dataset_val, dataloader_val = get_dataloader(self.cfg.data_root, self.cfg.dataset, if_train=False, batch_size=self.cfg.val_batch_size, num_workers=self.cfg.dataloader_workers, 
                            data_type='clip', clip_length=self.cfg.num_timesteps, use_default_collate=True, tokenizer=None, shuffle=False, if_return_bbox_im=self.cfg.load_bbox_image)
        self.dataset_val = dataset_val
        self.dataloader_val = dataloader_val

        if self.disable_images:
            dataset_val.disable_get_image()

        return dataloader_val



