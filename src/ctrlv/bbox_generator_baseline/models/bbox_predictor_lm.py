import torch 
import torch.nn.functional as F 
from torch import nn 
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm
from PIL import Image

from ctrlv.bbox_generator_baseline.modules import Encoder, Decoder, EncoderCoords, DecoderCoords
from ctrlv.bbox_generator_baseline.utils import create_lambda_lr, process_data

torch.set_printoptions(sci_mode=True)


def check_nan(tensor):
    return torch.isnan(tensor).any().item()

class BboxPredictorLM(pl.LightningModule):
    def __init__(self, cfg):
        super(BboxPredictorLM, self).__init__()

        self.cfg = cfg
        self.frame_size = (self.cfg.train_W, self.cfg.train_H)
        self.gt_frame_size = (1242, 375) # NOTE: Change for datasets other than Kitti
        if self.cfg.dataset == 'nuscenes':
            self.gt_frame_size = (1600, 900)
            
        if not self.cfg.pred_coords:
            self.cfg.vocabulary_size = self.cfg.dir_disc * self.cfg.norm_disc

        self.encoder = Encoder(self.cfg) if not self.cfg.pred_coords else EncoderCoords(self.cfg)
        self.decoder = Decoder(self.cfg) if not self.cfg.pred_coords else DecoderCoords(self.cfg)

        
    def forward_predict(self, data_dict, init_images=None, encoder_out=None):
        # Performing autoregressive rollout: only compute encoder_out once

        if encoder_out is None:
            encoder_out = self.encoder(data_dict, init_images)
            
        if encoder_out is None: return None, None

        decoder_out = self.decoder(encoder_out)

        return decoder_out, encoder_out


    def forward(self, agent_data, init_images=None):
        data_dict = process_data(self.cfg, agent_data, bbox_frame_size=self.gt_frame_size)

        encoder_out = self.encoder(data_dict, init_images)
        if encoder_out is None: return None, None

        decoder_out = self.decoder(encoder_out)

        return decoder_out, encoder_out
            

    def compute_loss_batch(self, agent_data, init_images=None):

        decoder_out, encoder_out = self(agent_data, init_images)
        if decoder_out is None: 
            print("Skipping training step: all batches were invalid")
            return None, None  # Skip step if data was invalid
        
        if not self.cfg.pred_coords:
            # For action predictions
            action_preds = decoder_out['action_preds']
            action_targets = encoder_out['actions_tokenized']
        else:
            # For coordinate predictions
            action_preds = decoder_out['coords_preds']
            action_targets = encoder_out['coords_tokenized']
            existence_mask = encoder_out['existence']
            if self.cfg.existence_head:
                existence_preds = decoder_out['existence_preds']
        
        batch_size, num_timesteps, num_agents, num_actions, _ = action_preds.shape
        vocab_size = self.cfg.vocabulary_size if not self.cfg.regression else 1

        # Align correctly
        action_preds = action_preds[:, :-1]  # [batch_size, (num_timesteps - 1), num_agents, num_actions, VOCABULARY_SIZE]
        action_targets = action_targets[:, 1:]

        # [batch_size * (num_timesteps - 1) * num_agents, VOCABULARY_SIZE, num_actions]
        action_preds = action_preds.reshape(batch_size * (num_timesteps - 1) * num_agents, num_actions, vocab_size).permute(0, 2, 1)

        # [batch_size * (num_timesteps - 1) * num_agents, num_actions]
        action_targets = action_targets.reshape(batch_size * (num_timesteps - 1) * num_agents, num_actions)

        if check_nan(action_preds) or check_nan(action_targets):
            print("Nan values in preds or targets")
        
        loss_components = {}
        if not self.cfg.pred_coords:
            loss = F.cross_entropy(action_preds, action_targets.long(), reduction='none')

            existence_mask = encoder_out['existence_mask']
            existence_mask = existence_mask[:, 1:].reshape(-1, 1)
            loss *= existence_mask.float()
            # Take mean loss over batch (according to existence mask)
            loss = loss.sum() / (existence_mask.sum() * num_actions)

            # NOTE: Don't think we need to ignore actions for which the states were given in the decoder... (model still needs to learn to map bboxes to actions)
            # if self.cfg.initial_frames_condition_num > 1:
            #     initial_actions_mask = torch.ones_like(decoder_out['existence_mask'])
            #     initial_actions_mask[:, :(self.cfg.initial_frames_condition_num - 1)] = 0
            #     loss *= initial_actions_mask.float()
        else:

            if self.cfg.regression:
                action_preds = action_preds.squeeze(-2) # Vocab size is 1, so we remove this dimension

            if self.cfg.existence_head:
                existence_preds = existence_preds[:, :-1]
                existence_mask = existence_mask[:, 1:].squeeze(-1)
                existence_targets = existence_mask.to(torch.float)

                existence_loss = F.binary_cross_entropy_with_logits(existence_preds, existence_targets, reduction='mean')
                existence_loss = (self.cfg.existence_loss_weight * existence_loss)

                # Mask out predictions that are supposed to be null
                existence_mask = existence_mask.flatten()
                action_preds = action_preds[existence_mask]
                action_targets = action_targets[existence_mask]

            if not self.cfg.regression:
                coords_loss = F.cross_entropy(action_preds, action_targets, reduction='mean') # Take mean loss over batch
            else:
                coords_loss = F.mse_loss(action_preds, action_targets, reduction='mean')
            
            coords_loss *= self.cfg.coords_loss_weight

            loss = coords_loss + existence_loss

            loss_components['coords_loss'] = coords_loss
            loss_components['existence_loss'] = existence_loss

        if check_nan(loss):
            print("Loss is nan")

        return loss, loss_components


    def training_step(self, data, batch_idx):
        
        # from ctrlv.datasets import kitti_clip_collate_fn
        # dset = torch.utils.data.Subset(self.trainer.datamodule.dataset_train, list(range(285*16, 285*16 + 200)))
        # collate_fn = lambda x: kitti_clip_collate_fn(x, None)

        # subset_loader = torch.utils.data.DataLoader(
        #     dset,
        #     collate_fn=collate_fn,
        #     batch_size=16,
        #     num_workers=1,
        #     shuffle=False,
        #     pin_memory=True,
        #     drop_last=True
        # )

        # for data in subset_loader:
        
        init_images = self.get_init_images(data, train=True) if self.cfg.map_embedding else None
        loss, loss_components = self.compute_loss_batch(data['objects'], init_images)

        # Log info
        if loss is not None:
            self.log('loss', loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        
        if loss_components is not None:
            if loss_components.get('coords_loss'):
                self.log('coords_loss', loss_components.get('coords_loss'), prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)

            if loss_components.get('existence_loss'):
                self.log('existence_loss', loss_components.get('existence_loss'), prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)

        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', cur_lr, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)

        return loss

    
    def validation_step(self, data, batch_idx):
        batch_size, _, _, _ = data['objects']['bbox'].shape
        
        init_images = self.get_init_images(data, train=False)
        val_loss, loss_components = self.compute_loss_batch(data['objects'], init_images) 

        if val_loss is not None:
            self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

        if loss_components is not None:
            if loss_components.get('coords_loss'):
                self.log('coords_loss', loss_components.get('coords_loss'), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

            if loss_components.get('existence_loss'):
                self.log('existence_loss', loss_components.get('existence_loss'), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

    # def on_validation_epoch_end(self):
    #     if self.trainer.datamodule.dataloader_val is None:
    #         return
        
    #     demo_samples = get_n_training_samples(self.trainer.datamodule.dataloader_val, self.cfg.num_demo_samples)
    #     self.eval()
    #     log_dict = self.generate_rollouts(demo_samples)
    #     self.train()
    #     # self.log_dict(log_dict)
    

    def get_init_images(self, data, train=True):
        dataset = self.trainer.datamodule.dataset_train if train else self.trainer.datamodule.dataset_val
        
        sample_indices = data['indices']
        # batch_size = len(sample_indices)
        # device = data['objects']['bbox'].device
        init_images = []

        for i, clip_idx in enumerate(sample_indices):
            image_init_file = dataset.get_frame_file_by_index(clip_idx, 0)
            image_init = Image.open(image_init_file)

            # Make sure all the size match (this will be changed later for training size)
            image_init = image_init.resize((dataset.orig_W, dataset.orig_H)) 
            init_images.append(image_init)

        return init_images


    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms_encoder = grad_norm(self.encoder, norm_type=2)
        self.log_dict(norms_encoder)

        norms_decoder = grad_norm(self.decoder, norm_type=2)
        self.log_dict(norms_decoder)


    ### Taken largely from QCNet repository
    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        # assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.cfg.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=create_lambda_lr(self.cfg))

        return [optimizer], {"scheduler": scheduler,
                             "interval": "step",
                             "frequency": 1}



