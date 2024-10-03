import torch
import torch.nn.functional as F 
import numpy as np
from collections import defaultdict
from typing import Union

from ctrlv.bbox_prediction.models.bbox_predictor_lm import BboxPredictorLM
from ctrlv.bbox_prediction.utils import process_data, undiscretize_actions, actions_to_bbox_seq, coords_to_bbox_seq, create_video_from_numpy_array, undiscretize_coords, bbox_seq_to_coords, discretize_coords, bbox_seq_to_actions, discretize_actions
from ctrlv.utils import plot_3d_bbox, get_n_training_samples, get_dataloader
from ctrlv.metrics import binary_mask_iou

from torchvision import transforms
transform = transforms.Compose([
    transforms.Normalize(mean=(127.5, 127.5, 127.5), std=(127.5, 127.5, 127.5)), # map from [0, 255] to [-1,1]
 ])
torch.set_printoptions(sci_mode=False)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_nan(tensor):
    return torch.isnan(tensor).any().item()


class BboxPredictorLMPolicy:
    
    def __init__(self, cfg):
        self.cfg = cfg

        set_seed(cfg.seed)

        print("Loading model from path:", self.cfg.model_path)
        self.model = BboxPredictorLM.load_from_checkpoint(self.cfg.model_path, cfg=self.cfg)
        self.model.eval()
    

    def run(self, num_samples=1):
        dataset, dataloader_val = get_dataloader(self.cfg.data_root, self.cfg.dataset, if_train=False, batch_size=self.cfg.val_batch_size, num_workers=self.cfg.dataloader_workers, 
                            data_type='clip', clip_length=self.cfg.num_timesteps, use_default_collate=True, tokenizer=None, shuffle=self.cfg.shuffle_dataset, if_return_bbox_im=True, non_overlapping_clips=self.cfg.dataset_non_overlapping)
        
        # disable_images = not self.cfg.map_embedding
        # if self.cfg.disable_image_load:
        #     disable_images = True
        
        # if disable_images:
        #     dataset.disable_get_image()

        self.dataset = dataset
        self.frame_size = (dataset.train_W, dataset.train_H)
        # print("Dataset orig size:", dataset.orig_W, dataset.orig_H)
        self.gt_frame_size = (1242, 375) # NOTE: Change for datasets other than Kitti
        if self.cfg.dataset == 'nuscenes':
            self.gt_frame_size = (1600, 900)
        
        print(f"Fetching {num_samples} sample{'s' if num_samples != 1 else ''}...")
        demo_samples = get_n_training_samples(dataloader_val, num_samples, show_progress=(num_samples >= 10))

        _ = self.generate_rollouts(demo_samples)

    def compute_loss_batch(self, agent_data, init_images=None):

        decoder_out, encoder_out = self.model(agent_data, init_images)
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


    def get_bbox_seq_vid(self, data_dict, gt_bboxes=False):
        
        if not gt_bboxes:
            if not self.cfg.pred_coords:
                bboxes_pred = actions_to_bbox_seq(data_dict['actions'], data_dict['bboxes'][:, 0], discard_first_action=True)
            else:
                bboxes_pred = coords_to_bbox_seq(data_dict['coords'])
        else:
            # Just use bboxes that are given
            bboxes_pred = data_dict['bboxes']
            
        _, num_timesteps, num_agents, _ = bboxes_pred.shape

        if not self.cfg.pred_coords:
            # TODO: Determine existence of bboxes (?)
            initial_existence = data_dict['existence'][:, 0]
            initial_existence_mask = initial_existence.any(dim=2).squeeze() # Only consider bboxes that existed at the initial timestep
            bboxes_pred_masked = bboxes_pred[:, :, initial_existence_mask] 
            track_ids = torch.arange(num_agents)[initial_existence.any(dim=2).squeeze()]
            type_ids = data_dict['type_ids'][0, 0][initial_existence_mask].squeeze(-1)
        else:
            bboxes_pred_masked = bboxes_pred  # Consider all bboxes
            track_ids = torch.arange(num_agents)
            type_ids = data_dict['type_ids'][0, 0].squeeze(-1)

        # Scale bboxes to desired frame size
        bboxes_pred_masked[:, :, :, 0] *= self.frame_size[0]
        bboxes_pred_masked[:, :, :, 2] *= self.frame_size[0]
        bboxes_pred_masked[:, :, :, 1] *= self.frame_size[1]
        bboxes_pred_masked[:, :, :, 3] *= self.frame_size[1]

        # Generate bbox img frame
        img_stack = torch.zeros([num_timesteps, self.frame_size[1], self.frame_size[0], 3], device=bboxes_pred_masked.device)
        for t in range(num_timesteps):

            labels = []
            for i in range(bboxes_pred_masked.shape[2]):
                labels.append({
                    "bbox": bboxes_pred_masked[0, t, i],
                    "id_type": type_ids[i].item(),
                    "trackID": track_ids[i].item()
                })

            canvas = torch.zeros((3, self.frame_size[1], self.frame_size[0]))  # Black background
            bbox_im = plot_3d_bbox(canvas, labels, None, plot_2d_bbox=True)

            img_stack[t] = torch.tensor(bbox_im)
        
        dataset = self.dataset
        bbox_vid = transform(img_stack.permute(0, 3, 2, 1))
        bbox_vid_np = dataset.revert_transform_no_resize(bbox_vid).detach().cpu().numpy()*255
        bbox_vid_np = bbox_vid_np.astype(np.uint8).transpose([0, 3, 2, 1])

        bbox_img_stack_np = img_stack.permute(0, 3, 1, 2).detach().cpu().numpy()

        return bbox_vid_np, bbox_img_stack_np


    def generate_rollouts(self, demo_samples):

        log_dict = defaultdict(list)
        num_agents = self.cfg.max_num_agents
        num_timesteps = self.cfg.num_timesteps

        avg_maskIOU, avg_maskP, avg_maskR = [], [], []
        avg_first_last_IOU, avg_first_last_P, avg_first_last_R = [], [], []
        
        for sample_i, sample in enumerate(demo_samples):
            
            agent_data = sample['objects_tensors']
            init_images = [sample['image_init']] if self.cfg.map_embedding else None
            gt_data_dict = process_data(self.cfg, agent_data, bbox_frame_size=self.gt_frame_size)
            gt_data_dict = {k: v[0].unsqueeze(0) for k,v in gt_data_dict.items() if v is not None} # Only keep first batch

            # Compute loss
            loss, loss_components = self.compute_loss_batch(agent_data, init_images)
            if loss is not None:
                print(f"Sample {sample_i} loss:", loss.item())
        
            if loss_components is not None:
                if loss_components.get('coords_loss'):
                    print("Coords loss:", loss_components.get('coords_loss'))

                if loss_components.get('existence_loss'):
                    print("Existence loss:", loss_components.get('existence_loss'))
            
            target_token_name = 'actions' if not self.cfg.pred_coords else 'coords'

            # NOTE: Not sure why but we need to crop to `num_agents` when <30
            gt_data_dict[target_token_name] = gt_data_dict[target_token_name][:, :, 0:num_agents]
            gt_data_dict['bboxes'] = gt_data_dict['bboxes'][:, :, 0:num_agents]
            gt_data_dict['type_ids'] = gt_data_dict['type_ids'][:, :, 0:num_agents]
            gt_data_dict['existence'] = gt_data_dict['existence'][:, :, 0:num_agents]

            curr_data_dict = {
                target_token_name: torch.zeros_like(gt_data_dict[target_token_name]),
                'bboxes': torch.zeros_like(gt_data_dict['bboxes']),
                'type_ids': torch.zeros_like(gt_data_dict['type_ids']),
                'existence': torch.zeros_like(gt_data_dict['existence'])
            }
            n_frames = self.cfg.initial_frames_condition_num
            curr_data_dict[target_token_name][:, :n_frames] = gt_data_dict[target_token_name][:, :n_frames]    # [batch_size=1, num_timesteps, num_agents, 2, 2] or [batch_size=1, num_timesteps, num_agents, 4]
            curr_data_dict['bboxes'][:, :n_frames] = gt_data_dict['bboxes'][:, :n_frames]      # Only used for context at first timestep in encoder
            curr_data_dict['type_ids'][:, :n_frames] = gt_data_dict['type_ids'][:, :n_frames]  # Only used for context at first timestep in encoder
            # curr_data_dict['existence'][:, :n_frames] = gt_data_dict['existence'][:, :n_frames]

            if self.cfg.condition_last_frame:
                curr_data_dict[target_token_name][:, -1] = gt_data_dict[target_token_name][:, -1]    # [batch_size=1, num_timesteps, num_agents, 2, 2] or [batch_size=1, num_timesteps, num_agents, 4]
                curr_data_dict['bboxes'][:, -1] = gt_data_dict['bboxes'][:, -1]      # Only used for context at first timestep in encoder
                curr_data_dict['type_ids'][:, -1] = gt_data_dict['type_ids'][:, -1]  # Only used for context at first timestep in encoder
                # curr_data_dict['existence'][:, :n_frames] = gt_data_dict['existence'][:, -1]

            # NOTE: Currently only configured to consider initial agents only, and assuming they will always exist throughout the episode
            # if self.cfg.only_keep_initial_agents:
            curr_data_dict['existence'] = gt_data_dict['existence'][:, 0, :, :].repeat(1, num_timesteps, 1, 1)
            curr_data_dict['type_ids'] = gt_data_dict['type_ids']  # Useful when using state_embeddings in decoder: we need type_id at each timestep

            encoder_out = None
            invalid_batches = False
            for t in range(num_timesteps):

                # Warmup for frames used as conditioning before making predictions
                if t < self.cfg.initial_frames_condition_num:
                    continue

                # Recompute the encoder every time if we need to compute state embeddings (which are updated on the fly).
                #   No need to recompute embeddings if using teacher-forcing, because they will all have been done on the first pass
                passed_encoder_out = encoder_out if (not self.cfg.use_state_embeddings or self.cfg.teacher_force_eval) else None  

                forward_input_dict = curr_data_dict if not self.cfg.teacher_force_eval else gt_data_dict
                decoder_out, encoder_out = self.model.forward_predict(forward_input_dict, init_images=init_images, encoder_out=passed_encoder_out)

                if decoder_out is None: 
                    print("Skipping eval step: all batches were invalid")
                    invalid_batches = True
                    break
                
                target_pred_name = 'action_preds' if not self.cfg.pred_coords else 'coords_preds'
                action_preds = decoder_out[target_pred_name] # [batch_size=1, num_timesteps, num_agents, 2 or 4, VOCABULARY_SIZE]
                num_preds = action_preds.shape[3]

                if not self.cfg.regression:
                    next_actions = []
                    for i in range(num_preds):
                        next_action_logits = action_preds[0, t-1, :, i]  # [num_agents, VOCABULARY_SIZE]
                        next_action_dist = F.softmax(next_action_logits / self.cfg.action_temp, dim=-1)
                        next_action = torch.multinomial(next_action_dist, 1) # [num_agents, 1]
                        next_actions.append(next_action)
                    
                    curr_action = torch.cat(next_actions, dim=-1).unsqueeze(0).unsqueeze(1) # [batch_size=1, num_timesteps=1, num_agents, 2 or 4]
                else:
                    curr_action = action_preds[0, t-1, :].unsqueeze(0).unsqueeze(1).squeeze(-1) # [batch_size=1, num_timesteps=1, num_agents, 4]      

                if self.cfg.pred_coords:
                    existence_preds = decoder_out['existence_preds']
                    next_existence_logits = existence_preds[0, t-1]
                    next_existence = torch.sigmoid(next_existence_logits)
                    next_existence_mask = (next_existence > 0.5)

                if not self.cfg.pred_coords:
                    # Undiscretize actions
                    undisc_action = undiscretize_actions(curr_action, dir_disc=self.cfg.dir_disc, norm_disc=self.cfg.norm_disc)  # [batch_size=1, num_timesteps=1, num_agents, 2, 2]
                else:
                    curr_action *= next_existence_mask.reshape(1, 1, num_agents, 1)  # Set actions to 0 for the bboxes that should not be rendered (bboxes predicted as null) 

                    if not self.cfg.regression:
                        undisc_action = undiscretize_coords(self.cfg, curr_action)  # [batch_size=1, num_timesteps=1, num_agents, 4]  
                    else:
                        undisc_action = curr_action


                # Update dicts with new actions
                action_embeddings = self.model.encoder.embed_tokenized_actions(curr_action) if not self.cfg.regression else self.model.encoder.embed_coords_regression(curr_action)
                curr_data_dict[target_token_name][:, t] = undisc_action  # Used to render bboxes
                if t < num_timesteps and not self.cfg.teacher_force_eval:  # Align correctly (last one is not needed)
                    target_embeddings = 'action_embeddings' if not self.cfg.pred_coords else 'coords_embeddings'
                    encoder_out[target_embeddings][:, t] = action_embeddings  # Used in decoder

                # Compute next bboxes: useful when using state_embeddings in decoder
                if not self.cfg.pred_coords:
                    next_bbox = actions_to_bbox_seq(undisc_action, curr_data_dict['bboxes'][:, t-1])
                    curr_data_dict['bboxes'][:, t] = next_bbox
            
            if invalid_batches:
                continue
            
            # Compute bbox sequence for predictions
            bbox_vid_np, bbox_img_stack_np = self.get_bbox_seq_vid(curr_data_dict)

            # Compute metrics
            gt_bbox_vid_np, gt_bbox_img_stack_np = self.get_bbox_seq_vid(gt_data_dict, gt_bboxes=False)
            maskIOU, maskP, maskR = binary_mask_iou(bbox_img_stack_np, gt_bbox_img_stack_np)
            first_last_IOU, first_last_P, first_last_R = binary_mask_iou(bbox_img_stack_np[[0, -1], :, :, :], gt_bbox_img_stack_np[[0, -1], :, :, :])

            avg_maskIOU.append(maskIOU)
            avg_maskP.append(maskP)
            avg_maskR.append(maskR)
            avg_first_last_IOU.append(first_last_IOU)
            avg_first_last_P.append(first_last_P)
            avg_first_last_R.append(first_last_R)
            print(f"""IOU: {maskIOU:.4f}, Precision: {maskP:.4f}, Recall: {maskR:.4f} || (First and last frames) IOU: {first_last_IOU:.4f}, Precision: {first_last_P:.4f}, Recall: {first_last_R:.4f}\n""")

            # Save to drive
            if self.cfg.eval_videos_every > 0 and sample_i % self.cfg.eval_videos_every == 0:
                create_video_from_numpy_array(bbox_vid_np, f"video_out/bbox_pred_{sample_i}.mp4", fps=self.cfg.video_fps)

                # GT video from dict
                # gt_bbox_2d_vid, _ = self.get_bbox_seq_vid(gt_data_dict)
                create_video_from_numpy_array(gt_bbox_vid_np, f"video_out/gt_dict_{sample_i}.mp4", fps=self.cfg.video_fps) 

                # GT bboxes
                create_video_from_numpy_array(sample['bbox_img_np'].transpose([0, 2, 3, 1]), f"video_out/gt_bbox_{sample_i}.mp4", fps=self.cfg.video_fps) 

                # GT video
                create_video_from_numpy_array(sample['gt_clip_np'].transpose([0, 2, 3, 1]), f"video_out/gt_video_{sample_i}.mp4", fps=self.cfg.video_fps) 

                # GT reconstruction:
                # coords_gt = bbox_seq_to_coords(gt_data_dict['bboxes'])
                # coords_gt_disc = discretize_coords(self.cfg, coords_gt)
                # undisc_coords = undiscretize_coords(self.cfg, coords_gt_disc)
                # curr_data_dict['coords'] = undisc_coords
                
                # reconstructed_gt_bbox_vid, _ = self.get_bbox_seq_vid(curr_data_dict)
                # create_video_from_numpy_array(reconstructed_gt_bbox_vid, f"video_out/gt_reconstructed_{sample_i}.mp4", fps=self.cfg.video_fps) # GT video reconstructed
                
                actions_gt = bbox_seq_to_actions(gt_data_dict['bboxes'])
                actions_gt_disc = discretize_actions(actions_gt)
                undisc_action = undiscretize_actions(actions_gt_disc)
                curr_data_dict['actions'] = undisc_action
                
                reconstructed_gt_bbox_vid, _ = self.get_bbox_seq_vid(curr_data_dict)
                create_video_from_numpy_array(reconstructed_gt_bbox_vid, f"video_out/gt_reconstructed_{sample_i}.mp4", fps=self.cfg.video_fps) # GT video reconstructed
                
                # Log to wandb
                # log_dict[f"predicted_rollout_{sample_i}"].append(wandb.Video(bbox_vid_np, fps=self.cfg.video_fps))
                # log_dict[f"gt_bbox_frames_{sample_i}"].append(wandb.Video(sample['bbox_img_np'].transpose([0, 2, 3, 1]), fps=self.cfg.video_fps))

        print(f"""\n[Average metrics ({len(avg_maskIOU)} sample{'s' if len(avg_maskIOU) != 1 else ''})] IOU: {np.array([avg_maskIOU]).mean():.4f}, Precision: {np.array([avg_maskP]).mean():.4f}, Recall: {np.array([avg_maskR]).mean():.4f} || (First and last frames) IOU: {np.array([avg_first_last_IOU]).mean():.4f}, Precision: {np.array([avg_first_last_P]).mean():.4f}, Recall: {np.array([avg_first_last_R]).mean():.4f}""")
        print(f"""\n[Metrics std     ({len(avg_maskIOU)} sample{'s' if len(avg_maskIOU) != 1 else ''})] IOU: {np.array([avg_maskIOU]).std():.4f}, Precision: {np.array([avg_maskP]).std():.4f}, Recall: {np.array([avg_maskR]).std():.4f} || (First and last frames) IOU: {np.array([avg_first_last_IOU]).std():.4f}, Precision: {np.array([avg_first_last_P]).std():.4f}, Recall: {np.array([avg_first_last_R]).std():.4f}""")
        
        print("\nCopy-friendly format:")
        print(f"{np.array([avg_maskIOU]).mean():.4f},{np.array([avg_maskP]).mean():.4f},{np.array([avg_maskR]).mean():.4f},{np.array([avg_first_last_IOU]).mean():.4f},{np.array([avg_first_last_P]).mean():.4f},{np.array([avg_first_last_R]).mean():.4f}")
        print(f"{np.array([avg_maskIOU]).std():.4f},{np.array([avg_maskP]).std():.4f},{np.array([avg_maskR]).std():.4f},{np.array([avg_first_last_IOU]).std():.4f},{np.array([avg_first_last_P]).std():.4f},{np.array([avg_first_last_R]).std():.4f}")
        return log_dict
    