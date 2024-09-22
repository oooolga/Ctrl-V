import torch
import torch.nn as nn

from ctrlv.bbox_prediction.utils import weight_init, MLPLayer, PositionalEncoding, discretize_actions, ImageEncoder


class Encoder(nn.Module):

    def __init__(self, cfg):
        super(Encoder, self).__init__()
        
        self.cfg = cfg
        hidden_dim = self.cfg.hidden_dim

        self.embed_action = nn.Embedding(int(self.cfg.vocabulary_size), hidden_dim)
        self.embed_action_combine = nn.Linear(hidden_dim * 2, hidden_dim)

        self.embed_state = MLPLayer(self.cfg.state_dim + 1, hidden_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout=0, max_len=self.cfg.num_timesteps)  # NOTE: Consider increasing max_len to accomodate for longer sequences during inference
        self.embed_agent_id = nn.Embedding(self.cfg.max_num_agents, hidden_dim)

        # NOTE: Could also consider encoding 3D object dimensions, location and rotation_y as initial state information

        # Initial image frame is embedding and used as "map context information"
        if self.cfg.map_embedding:
            self.image_encoder = ImageEncoder(cfg)

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, 
                                                               nhead=self.cfg.num_heads,
                                                               dim_feedforward=self.cfg.dim_feedforward,
                                                               batch_first=True)                                            
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=self.cfg.num_encoder_layers)

        self.apply(weight_init)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    def embed_tokenized_actions(self, actions_tokenized):
        # actions_tokenized: [batch_size, num_timesteps, num_agents, 2]
        action1_embeddings = self.embed_action(actions_tokenized[:, :, :, 0])
        action2_embeddings = self.embed_action(actions_tokenized[:, :, :, 1])
        action_embeddings = torch.cat([action1_embeddings, action2_embeddings], dim=-1)
        action_embeddings = self.embed_action_combine(action_embeddings) 
        
        return action_embeddings

    def forward(self, data_dict, init_images=None):
        num_agents = self.cfg.max_num_agents
        
        actions = data_dict['actions'][:, :, 0:num_agents].to(self.device)      # [batch_size, num_timesteps, num_agents, 2, 2]
        bboxes = data_dict['bboxes'][:, :, 0:num_agents].to(self.device)        # [batch_size, num_timesteps, num_agents, 4]
        type_ids = data_dict['type_ids'][:, :, 0:num_agents].to(self.device)    # [batch_size, num_timesteps, num_agents]
        existence = data_dict['existence'][:, :, 0:num_agents].to(self.device)  # [batch_size, num_timesteps, num_agents]

        batch_size, num_timesteps, num_agents, _ = bboxes.shape

        if self.cfg.last_frame_traj:
            # Replace last bbox frame with "trajectory frame" (ie: center of bbox), which we will represent as [x, y, 0, 0] for dims to match
            x1, y1, x2, y2 = bboxes[:, -1, :, 0], bboxes[:, -1, :, 1], bboxes[:, -1, :, 2], bboxes[:, -1, :, 3]
            bboxes[:, -1, :, 0] = (torch.max(x1, x2) + torch.min(x1, x2)) / 2
            bboxes[:, -1, :, 1] = (torch.max(y1, y2) + torch.min(y1, y2)) / 2
            bboxes[:, -1, :, 2:] = 0.0

        # Concatenate the bbox with the agent type for the input "state"
        states = torch.cat([bboxes, type_ids], dim=-1)
        state_embeddings = self.embed_state(states)

        agent_ids = torch.arange(num_agents).unsqueeze(0).to(bboxes.device)
        id_embeddings = self.embed_agent_id(agent_ids).unsqueeze(1)

        timesteps = torch.arange(num_timesteps).unsqueeze(0)
        timestep_embeddings = self.positional_encoding(timesteps).unsqueeze(2)

        # Embed actions
        # TODO: Currently not correctly handling existence of actions when agent comes into or out of existence.
        #       Since we are reporting actions into and out from null states [0, 0, 0, 0] (ie: ([x1, y1, x1, y2] -> [0, 0, 0, 0])), 
        #       these should be masked because they are not real actions.
        actions_tokenized = discretize_actions(actions, dir_disc=self.cfg.dir_disc, norm_disc=self.cfg.norm_disc).to(torch.int)  # TODO: Could replace first action (which is constant) by SOS token explicitely (but perhaps it is already playing this role)
        action_embeddings = self.embed_tokenized_actions(actions_tokenized)

        # Combine embeddings
        state_embeddings = state_embeddings + action_embeddings + id_embeddings + timestep_embeddings

        if self.cfg.only_keep_initial_agents:
            # Mask out agents completely if they were not present in first timestep
            initial_existence = existence[:, 0]

            if self.cfg.always_predict_initial_agents:
                existence = initial_existence.unsqueeze(1).repeat(1, num_timesteps, 1, 1)
            else:
                existence = torch.logical_and(existence, initial_existence.unsqueeze(1))  

        # Zero out embeddings at timesteps where the agent does not have visible bbox
        state_embeddings *= existence  

        num_initial_frames = self.cfg.initial_frames_condition_num or 1
        initial_state_embeddings = state_embeddings[:, :num_initial_frames]

        # # Only select batches where the initial frame has at least one bbox in existence
        # valid_batches = existence[:, 0].any(dim=1).squeeze()

        # Only select batches where there is at least one bbox in existence at every timestep
        valid_batches = existence.sum(dim=2).squeeze(-1).all(dim=1).squeeze()
        if torch.numel(valid_batches) == 1:
            valid_batches = valid_batches.unsqueeze(0) # Handle indexing error for 0-d tensor (when batch_size=1)
        action_embeddings = action_embeddings[valid_batches]
        state_embeddings = state_embeddings[valid_batches]
        initial_state_embeddings = initial_state_embeddings[valid_batches]
        existence = existence[valid_batches]
        bboxes = bboxes[valid_batches]
        actions_tokenized = actions_tokenized[valid_batches]
        if init_images is not None:
            init_images = [img for idx, img in enumerate(init_images) if valid_batches[idx]]

        if bboxes.shape[0] < batch_size:
            print(f"Dropping {batch_size - bboxes.shape[0]}/{batch_size} invalid batches")
        batch_size = bboxes.shape[0]

        if batch_size == 0:
            print("All batches are invalid, returning None for encoder")
            return None

        if not self.cfg.condition_last_frame:
            conditioning_existence = existence[:, :num_initial_frames]
            input_state_embeddings = initial_state_embeddings
        else:
            # For last frame conditioning
            conditioning_existence = torch.cat([existence[:, :num_initial_frames], existence[:, -1:]], dim=1)
            input_state_embeddings = torch.cat([initial_state_embeddings, state_embeddings[:, -1:]], dim=1)
        
        conditioning_existence = conditioning_existence.reshape(batch_size, -1, 1)
        src_key_padding_mask = torch.zeros_like(conditioning_existence.squeeze(-1), dtype=torch.bool)
        src_key_padding_mask[conditioning_existence.squeeze(-1) == False] = True  # 'True' values are the ones that get masked out

        input_state_embeddings = input_state_embeddings.reshape(batch_size, -1, self.cfg.hidden_dim)
        if self.cfg.map_embedding:
            # Encode map information with VAE, then combine with input_state_embeddings (and update src_key_padding_mask)
            init_image_encodings = self.image_encoder(init_images, image_size=(self.cfg.train_H, self.cfg.train_W))
            input_embeddings = torch.cat([input_state_embeddings, init_image_encodings], dim=1)
            valid_mask = torch.zeros([init_image_encodings.shape[0], init_image_encodings.shape[1]], device=input_embeddings.device).bool() # No masking for init_image
            src_key_padding_mask = torch.cat([src_key_padding_mask, valid_mask], dim=1)
        else:
            input_embeddings = input_state_embeddings

        encoder_embeddings = self.transformer_encoder(input_embeddings, src_key_padding_mask=src_key_padding_mask)
        encoder_embeddings[:, :conditioning_existence.shape[1]] *= conditioning_existence

        existence_mask = existence.reshape(batch_size, num_timesteps * num_agents)
        tgt_key_padding_mask = torch.zeros_like(existence_mask, dtype=torch.bool)
        tgt_key_padding_mask[existence_mask == False] = True

        if torch.isnan(encoder_embeddings).any().item():
            print("Nan values in embeddings")

        return {
            'encoder_embeddings': encoder_embeddings,
            'action_embeddings': action_embeddings,
            'state_embeddings': state_embeddings,
            'actions_tokenized': actions_tokenized,
            'src_key_padding_mask': src_key_padding_mask,
            'tgt_key_padding_mask': tgt_key_padding_mask,
            'existence_mask': existence,
            'id_embeddings': id_embeddings,
            'timestep_embeddings': timestep_embeddings,
            'valid_batch_mask': valid_batches
        }
