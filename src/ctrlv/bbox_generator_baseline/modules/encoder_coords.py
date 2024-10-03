import torch
import torch.nn as nn

from ctrlv.bbox_generator_baseline.utils import weight_init, MLPLayer, PositionalEncoding, discretize_coords, ImageEncoder


class EncoderCoords(nn.Module):

    def __init__(self, cfg):
        super(EncoderCoords, self).__init__()
        
        self.cfg = cfg
        hidden_dim = self.cfg.hidden_dim

        if not self.cfg.regression:
            self.embed_coords = nn.Embedding(self.cfg.vocabulary_size, hidden_dim)
            self.embed_coords_combine = nn.Linear(hidden_dim * 4, hidden_dim)
        else:
            self.embed_coords = MLPLayer(self.cfg.state_dim, hidden_dim, hidden_dim)

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
    

    def embed_tokenized_actions(self, coords_tokenized):
        # coords_tokenized: [batch_size, num_timesteps, num_agents, 4]

        coord1_embeddings = self.embed_coords(coords_tokenized[:, :, :, 0])
        coord2_embeddings = self.embed_coords(coords_tokenized[:, :, :, 1])
        coord3_embeddings = self.embed_coords(coords_tokenized[:, :, :, 2])
        coord4_embeddings = self.embed_coords(coords_tokenized[:, :, :, 3])

        coords_embeddings = torch.cat([coord1_embeddings, coord2_embeddings, coord3_embeddings, coord4_embeddings], dim=-1)
        coords_embeddings = self.embed_coords_combine(coords_embeddings) 
        
        return coords_embeddings

    def embed_coords_regression(self, coords):
        coords = torch.clip(coords, min=0.0, max=1.0)
        coords_embeddings = self.embed_coords(coords)

        return coords_embeddings

    def forward(self, data_dict, init_images=None):
        num_agents = self.cfg.max_num_agents
        
        coords = data_dict['coords'][:, :, 0:num_agents].to(self.device)        # [batch_size, num_timesteps, num_agents, 4]
        type_ids = data_dict['type_ids'][:, :, 0:num_agents].to(self.device)    # [batch_size, num_timesteps, num_agents]
        existence = data_dict['existence'][:, :, 0:num_agents].to(self.device)  # [batch_size, num_timesteps, num_agents]

        batch_size, num_timesteps, num_agents, _ = coords.shape

        if self.cfg.last_frame_traj:
            # Replace last bbox frame with "trajectory frame" (ie: center of bbox), which we will represent as [x, y, 0, 0] for dims to match
            x1, y1, x2, y2 = coords[:, -1, :, 0], coords[:, -1, :, 1], coords[:, -1, :, 2], coords[:, -1, :, 3]
            coords[:, -1, :, 0] = (torch.max(x1, x2) + torch.min(x1, x2)) / 2
            coords[:, -1, :, 1] = (torch.max(y1, y2) + torch.min(y1, y2)) / 2
            coords[:, -1, :, 2:] = 0.0

        # Concatenate the bbox with the agent type for the input "state"
        states = torch.cat([coords, type_ids], dim=-1)
        state_embeddings = self.embed_state(states)

        agent_ids = torch.arange(num_agents).unsqueeze(0).to(coords.device)
        id_embeddings = self.embed_agent_id(agent_ids).unsqueeze(1)

        timesteps = torch.arange(num_timesteps).unsqueeze(0).to(coords.device)
        timestep_embeddings = self.positional_encoding(timesteps).unsqueeze(2)

        if not self.cfg.regression:
            coords_tokenized = discretize_coords(self.cfg, coords)
            coords_embeddings = self.embed_tokenized_actions(coords_tokenized)  # "Tokenized coords embedding"
        else:
            coords_embeddings = self.embed_coords_regression(coords)
            coords_tokenized=coords # Not actually tokenized here, but they serve the same purpose

        # Combine embeddings
        state_embeddings = state_embeddings + coords_embeddings + id_embeddings + timestep_embeddings

        num_initial_frames = self.cfg.initial_frames_condition_num or 1
        initial_state_embeddings = state_embeddings[:, :num_initial_frames]

        if not self.cfg.condition_last_frame:
            input_state_embeddings = initial_state_embeddings
        else:
            # For last frame conditioning
            input_state_embeddings = torch.cat([initial_state_embeddings, state_embeddings[:, -1:]], dim=1)
        
        input_state_embeddings = input_state_embeddings.reshape(batch_size, -1, self.cfg.hidden_dim)
        if self.cfg.map_embedding:
            # Encode map information with VAE, then combine with input_state_embeddings (and update src_key_padding_mask)
            init_image_encodings = self.image_encoder(init_images, image_size=(self.cfg.train_H, self.cfg.train_W))
            input_embeddings = torch.cat([input_state_embeddings, init_image_encodings], dim=1)
        else:
            input_embeddings = input_state_embeddings

        encoder_embeddings = self.transformer_encoder(input_embeddings, src_key_padding_mask=None)

        if torch.isnan(encoder_embeddings).any().item():
            print("Nan values in embeddings")

        return {
            'encoder_embeddings': encoder_embeddings,
            'coords_embeddings': coords_embeddings,
            'state_embeddings': state_embeddings,
            'coords_tokenized': coords_tokenized,
            'id_embeddings': id_embeddings,
            'timestep_embeddings': timestep_embeddings,
            'existence': existence
        }
