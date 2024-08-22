import torch
import torch.nn.functional as F 
import numpy as np
from collections import defaultdict
from typing import Union

from sd3d.bbox_prediction.models.bbox_predictor_lm import BboxPredictorLM
from sd3d.bbox_prediction.utils import process_data, undiscretize_actions, actions_to_bbox_seq, create_video_from_numpy_array, bbox_seq_to_actions, discretize_actions, VOCABULARY_SIZE
from sd3d.utils import plot_3d_bbox, get_n_training_samples, get_dataloader
from sd3d.metrics import binary_mask_iou

from torchvision import transforms
transform = transforms.Compose([
    transforms.Normalize(mean=(127.5, 127.5, 127.5), std=(127.5, 127.5, 127.5)), # map from [0, 255] to [-1,1]
 ])
torch.set_printoptions(sci_mode=False)


class BboxPredictorLMPolicy:
    
    def __init__(self, cfg):
        self.cfg = cfg

        print("Loading model from path:", self.cfg.model_path)
        self.model = BboxPredictorLM.load_from_checkpoint(self.cfg.model_path, cfg=self.cfg)
        self.model.eval()
    

    def run(self, num_samples=1):
        dataset, dataloader_val = get_dataloader(self.cfg.data_root, self.cfg.dataset, if_train=False, batch_size=self.cfg.val_batch_size, num_workers=self.cfg.dataloader_workers, 
                            data_type='clip', clip_length=self.cfg.num_timesteps, use_default_collate=True, tokenizer=None, shuffle=False, if_return_bbox_im=True, non_overlapping_clips=self.cfg.dataset_non_overlapping)
        
        self.dataset = dataset
        self.frame_size = (dataset.train_W, dataset.train_H)
        # print("Dataset orig size:", dataset.orig_W, dataset.orig_H)
        self.gt_frame_size = (dataset.orig_W, dataset.orig_H) # NOTE: Change for datasets other than Kitti
        
        print(f"Fetching {num_samples} sample{'s' if num_samples != 1 else ''}...")
        demo_samples = get_n_training_samples(dataloader_val, num_samples, show_progess=(num_samples >= 10))
        _ = self.generate_rollouts(demo_samples)


    def compute_loss_batch(self, agent_data, init_images=None):

        decoder_out, encoder_out = self.model(agent_data, init_images)
        if decoder_out is None: 
            print("Skipping training step: all batches were invalid")
            return None  # Skip step if data was invalid
        
        action_preds = decoder_out['action_preds']
        action_targets = encoder_out['actions_tokenized']
        existence_mask = encoder_out['existence_mask']

        batch_size, num_timesteps, num_agents, _, _ = action_preds.shape

        # Align correctly
        action_preds = action_preds[:, :-1]  # [batch_size, num_timesteps - 1, num_agents, num_actions, VOCABULARY_SIZE]
        action_targets = action_targets[:, 1:]
        existence_mask = existence_mask[:, 1:].reshape(-1, 1)

        # [batch_size * (num_timesteps - 1) * num_agents, VOCABULARY_SIZE, num_actions]
        action_preds = action_preds.reshape(batch_size * (num_timesteps - 1) * num_agents, 2, VOCABULARY_SIZE).permute(0, 2, 1)

        # [batch_size * (num_timesteps - 1) * num_agents, 2]
        action_targets = action_targets.reshape(batch_size * (num_timesteps - 1) * num_agents, 2).long()
        
        loss = F.cross_entropy(action_preds, action_targets, reduction='none')
        loss *= existence_mask.float()

        # Take mean loss over batch (according to existence mask)
        loss = loss.sum() / existence_mask.sum()

        return loss


    def get_bbox_seq_vid(self, data_dict):
        bboxes_pred = actions_to_bbox_seq(data_dict['actions'], data_dict['bboxes'][:, 0], self.frame_size, discard_first_action=True)
        _, num_timesteps, num_agents, _ = bboxes_pred.shape

        # TODO: Determine existence of bboxes (?)
        initial_existence = data_dict['existence'][:, 0]
        initial_existence_mask = initial_existence.any(dim=2).squeeze()
        bboxes_pred_masked = bboxes_pred[:, :, initial_existence_mask] # Only consider bboxes that existed at the initial timestep
        track_ids = torch.arange(num_agents)[initial_existence.any(dim=2).squeeze()]
        type_ids = data_dict['type_ids'][0, 0][initial_existence_mask].squeeze(-1)

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
            gt_data_dict = process_data(agent_data, out_frame_size=self.frame_size, bbox_frame_size=self.gt_frame_size)
            gt_data_dict = {k: v[0].unsqueeze(0) for k,v in gt_data_dict.items()} # Only keep first batch

            # Compute loss
            # loss = self.compute_loss_batch(agent_data, init_images)
            # print(f"Sample {sample_i} loss:", loss.item())

            # NOTE: Not sure why we need to crop to num_agents when <30
            gt_data_dict['actions']= gt_data_dict['actions'][:, :, 0:num_agents]
            gt_data_dict['bboxes'] = gt_data_dict['bboxes'][:, :, 0:num_agents]
            gt_data_dict['type_ids'] = gt_data_dict['type_ids'][:, :, 0:num_agents]
            gt_data_dict['existence'] = gt_data_dict['existence'][:, :, 0:num_agents]

            curr_data_dict = {
                "actions": torch.zeros_like(gt_data_dict['actions']),
                "bboxes": torch.zeros_like(gt_data_dict['bboxes']),
                "type_ids": torch.zeros_like(gt_data_dict['type_ids']),
                "existence": torch.zeros_like(gt_data_dict['existence'])
            }
            n_frames = self.cfg.initial_frames_condition_num
            curr_data_dict['actions'][:, :n_frames] = gt_data_dict['actions'][:, :n_frames]    # [batch_size=1, num_timesteps, num_agents, 2, 2]
            curr_data_dict['bboxes'][:, :n_frames] = gt_data_dict['bboxes'][:, :n_frames]      # Only used for context at first timestep in encoder
            curr_data_dict['type_ids'][:, :n_frames] = gt_data_dict['type_ids'][:, :n_frames]  # Only used for context at first timestep in encoder
            # curr_data_dict['existence'][:, :n_frames] = gt_data_dict['existence'][:, :n_frames]

            if self.cfg.condition_last_frame:
                curr_data_dict['actions'][:, -1] = gt_data_dict['actions'][:, -1]    # [batch_size=1, num_timesteps, num_agents, 2, 2]
                curr_data_dict['bboxes'][:, -1] = gt_data_dict['bboxes'][:, -1]      # Only used for context at first timestep in encoder
                curr_data_dict['type_ids'][:, -1] = gt_data_dict['type_ids'][:, -1]  # Only used for context at first timestep in encoder
                # curr_data_dict['existence'][:, :n_frames] = gt_data_dict['existence'][:, -1]

            # NOTE: Currently only configured to consider initial agents only, and assuming they will always exist throughout the episode
            # if self.cfg.only_keep_initial_agents:
            curr_data_dict['existence'] = gt_data_dict['existence'][:, 0, :, :].repeat(1, num_timesteps, 1, 1)
            curr_data_dict['type_ids'] = gt_data_dict['type_ids']  # Useful when using state_embeddings in decoder: we need type_id at each timestep

            encoder_out = None
            for t in range(num_timesteps):

                # Warmup for frames used as conditioning before making predictions
                if t < self.cfg.initial_frames_condition_num:
                    continue
                
                passed_encoder_out = encoder_out if not self.cfg.use_state_embeddings else None  # Recompute the encoder every time if we need to compute state embeddings (which are updated on the fly)
                decoder_out, encoder_out = self.model.forward_predict(curr_data_dict, init_images=init_images, encoder_out=passed_encoder_out)
                # decoder_out, encoder_out = self.model.forward_predict(gt_data_dict, encoder_out=passed_encoder_out) # For open loop testing

                action_preds = decoder_out['action_preds'] # [batch_size=1, num_timesteps, num_agents, 2, VOCABULARY_SIZE]

                next_action_logits1 = action_preds[0, t-1, :, 0]  # [num_agents, VOCABULARY_SIZE]
                next_action_dist1 = F.softmax(next_action_logits1 / self.cfg.action_temp, dim=-1)
                next_action1 = torch.multinomial(next_action_dist1, 1) # [num_agents, 1]

                next_action_logits2 = action_preds[0, t-1, :, 1] # [num_agents, VOCABULARY_SIZE]
                next_action_dist2 = F.softmax(next_action_logits2 / self.cfg.action_temp, dim=-1)
                next_action2 = torch.multinomial(next_action_dist2, 1) # [num_agents, 1]

                # Undiscretize actions
                curr_action = torch.cat([next_action1, next_action2], dim=-1).unsqueeze(0).unsqueeze(1) # [batch_size=1, num_timesteps=1, num_agents, 2]
                action_embeddings = self.model.encoder.embed_tokenized_actions(curr_action)
                undisc_action = undiscretize_actions(curr_action)  # [batch_size=1, num_timesteps=1, num_agents, 2, 2]

                # Update dicts with new actions
                curr_data_dict['actions'][:, t-1] = undisc_action  # Used to render bboxes
                if t < num_timesteps:  # Align correctly (last one is not needed)
                    encoder_out['action_embeddings'][:, t] = action_embeddings  # Used in decoder
                
                # Compute next bboxes: useful when using state_embeddings in decoder
                next_bbox = actions_to_bbox_seq(undisc_action, curr_data_dict['bboxes'][:, t-1], self.frame_size)
                curr_data_dict['bboxes'][:, t] = next_bbox
            
            # Compute bbox sequence for predictions
            bbox_vid_np, bbox_img_stack_np = self.get_bbox_seq_vid(curr_data_dict)

            # Compute metrics
            maskIOU, maskP, maskR = binary_mask_iou(bbox_img_stack_np, sample['bbox_img_np'])
            first_last_IOU, first_last_P, first_last_R = binary_mask_iou(bbox_img_stack_np[[0, -1], :, :, :], sample['bbox_img_np'][[0, -1], :, :, :])

            avg_maskIOU.append(maskIOU)
            avg_maskP.append(maskP)
            avg_maskR.append(maskR)
            avg_first_last_IOU.append(first_last_IOU)
            avg_first_last_P.append(first_last_P)
            avg_first_last_R.append(first_last_R)
            print(f"""IOU: {maskIOU:.4f}, Precision: {maskP}, Recall: {maskR:.4f} || (First and last frames) IOU: {first_last_IOU:.4f}, Precision: {first_last_P:.4f}, Recall: {first_last_R:.4f}\n""")

            # Save to drive
            if sample_i % 10 == 0:
                create_video_from_numpy_array(bbox_vid_np, f"video_out/bbox_pred_{sample_i}.mp4", fps=self.cfg.video_fps)

            # GT reconstruction:
            # actions_gt = bbox_seq_to_actions(gt_data_dict['bboxes'], self.frame_size)
            # actions_gt_disc = discretize_actions(actions_gt)
            # undisc_action = undiscretize_actions(actions_gt_disc)
            # curr_data_dict['actions'] = undisc_action
            
            # reconstructed_gt_bbox_vid = self.get_bbox_seq_vid(curr_data_dict)
            # create_video_from_numpy_array(reconstructed_gt_bbox_vid, f"video_out/gt_reconstructed_{sample_i}.mp4", fps=self.cfg.video_fps) # GT video reconstructed

            # gt_bbox_2d_vid = self.get_bbox_seq_vid(gt_data_dict)
            # create_video_from_numpy_array(gt_bbox_2d_vid, f"video_out/gt_bbox_2d_{sample_i}.mp4", fps=self.cfg.video_fps) # GT video reconstructed
            
            # create_video_from_numpy_array(sample['bbox_img_np'].transpose([0, 2, 3, 1]), f"video_out/gt_bbox_{sample_i}.mp4", fps=self.cfg.video_fps) # GT bboxes
            # create_video_from_numpy_array(sample['gt_clip_np'].transpose([0, 2, 3, 1]), f"video_out/gt_video_{sample_i}.mp4", fps=self.cfg.video_fps) # GT video
            
            # Log to wandb
            # log_dict[f"predicted_rollout_{sample_i}"].append(wandb.Video(bbox_vid_np, fps=self.cfg.video_fps))
            # log_dict[f"gt_bbox_frames_{sample_i}"].append(wandb.Video(sample['bbox_img_np'].transpose([0, 2, 3, 1]), fps=self.cfg.video_fps))

        print(f"""\n[Average metrics ({len(avg_maskIOU)} sample{'s' if len(avg_maskIOU) != 1 else ''})] IOU: {np.array([avg_maskIOU]).mean():.4f}, Precision: {np.array([avg_maskP]).mean():.4f}, Recall: {np.array([avg_maskR]).mean():.4f} || (First and last frames) IOU: {np.array([avg_first_last_IOU]).mean():.4f}, Precision: {np.array([avg_first_last_P]).mean():.4f}, Recall: {np.array([avg_first_last_R]).mean():.4f}""")
        
        return log_dict