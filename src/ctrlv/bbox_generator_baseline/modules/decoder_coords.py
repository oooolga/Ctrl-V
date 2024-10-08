import torch
import torch.nn as nn
from torch.nn import Transformer

from ctrlv.bbox_generator_baseline.utils import weight_init, MLPLayer


class DecoderCoords(nn.Module):

    def __init__(self, cfg):
        super(DecoderCoords, self).__init__()

        self.cfg = cfg
        hidden_dim = self.cfg.hidden_dim

        self.embedding_layer_norm = nn.LayerNorm(hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim,
                                                   dim_feedforward=self.cfg.dim_feedforward,
                                                   nhead=self.cfg.num_heads,
                                                   batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.cfg.num_decoder_layers)

        # MLP to predict the next bbox state (4 tokens; 2 for each 2 corners)
        # TODO: Try two heads (one for each action)
        if not self.cfg.regression:
            self.predict_coords = MLPLayer(hidden_dim, hidden_dim, self.cfg.vocabulary_size * 4)
        else:
            self.predict_coords = MLPLayer(hidden_dim, hidden_dim, 4)
        
        if self.cfg.existence_head:
            # Separate head for predicting the existence of every bbox at every timestep
            self.predict_existence = MLPLayer(hidden_dim, hidden_dim, 1)

        self.causal_mask = self.get_causal_mask()

        # Init weights
        self.apply(weight_init)

    
    def get_causal_mask(self):
        num_agents = self.cfg.max_num_agents
        num_steps = self.cfg.num_timesteps
        state_index = 0

        num_types = 1
        num_tokens = num_agents * num_steps * num_types
        
        mask = Transformer.generate_square_subsequent_mask(num_tokens)
        multi_agent_mask = torch.Tensor(mask.shape).fill_(0)
        offset = 0
        index = 0
        for index in range(len(multi_agent_mask)):
            mask_out = torch.Tensor(num_agents * num_types).fill_(float('-inf'))
            agent_id = (index // num_types) % num_agents 
            mask_out[agent_id*num_types:(agent_id+1)*(num_types)] = 0
            multi_agent_mask[index, offset:offset+(num_agents * num_types)] = mask_out 
            
            if (index + 1) % (num_agents * num_types) == 0:
                offset += num_agents * num_types

        mask = torch.minimum(mask, multi_agent_mask)

        # current state of all agents is visible
        for index_i in range(len(mask)):
            timestep_idx = index_i // (num_types * num_agents)
            for index_j in range(len(mask)):
                if index_j < (timestep_idx + 1) * (num_agents*num_types) and index_j % num_types == state_index:
                    mask[index_i, index_j] = 0.

        return mask


    def forward(self, encoder_out):
        encoder_embeddings = encoder_out['encoder_embeddings']      # [batch_size, num_cond_frames * num_agents, hidden_dim]
        coords_embeddings = encoder_out['coords_embeddings']        # [batch_size, num_timesteps, num_agents, hidden_dim]
        id_embeddings = encoder_out['id_embeddings']                # [1, 1, num_agents, hidden_dim]
        timestep_embeddings = encoder_out['timestep_embeddings']    # [1, num_timesteps, 1, hidden_dim]

        batch_size, num_timesteps, num_agents, _ = coords_embeddings.shape

        # Reshape (flatten agent & timestep dims together, agent first)
        if not self.cfg.use_state_embeddings:
            coords_embeddings = coords_embeddings + id_embeddings + timestep_embeddings
            coords_embeddings = coords_embeddings.reshape(batch_size, num_timesteps * num_agents, self.cfg.hidden_dim)
            coords_embeddings = self.embedding_layer_norm(coords_embeddings) # NOTE: Might not be necessary?
            input_embeddings = coords_embeddings
        else:
            # TODO: Should probably not include map encodings in the state
            state_embeddings = encoder_out['state_embeddings']
            # print(state_embeddings[0, :, 1, :5])
            state_embeddings = state_embeddings.reshape(batch_size, num_timesteps * num_agents, self.cfg.hidden_dim)
            state_embeddings = self.embedding_layer_norm(state_embeddings) # NOTE: Might not be necessary?
            input_embeddings = state_embeddings
        
        
        decoder_out = self.transformer_decoder(input_embeddings, 
                                               encoder_embeddings, 
                                               tgt_mask=self.causal_mask.to(input_embeddings.device),
                                               tgt_key_padding_mask=None, 
                                               memory_key_padding_mask=None)
        
        coords_preds = self.predict_coords(decoder_out)

        existence_preds = None
        if self.cfg.existence_head:
            existence_preds = self.predict_existence(decoder_out)
            existence_preds = existence_preds.reshape(batch_size, num_timesteps, num_agents)

        if not self.cfg.regression:
            # [batch_size, num_timesteps, num_agents, 4, VOCABULARY_SIZE]
            coords_preds = coords_preds.reshape(batch_size, num_timesteps, num_agents, 4, self.cfg.vocabulary_size) 
        else:
            # [batch_size, num_timesteps, num_agents, 4, 1]
            coords_preds = coords_preds.reshape(batch_size, num_timesteps, num_agents, 4, 1) 

        if torch.isnan(coords_preds).any().item():
            print("Nan values in preds")

        return {
            'coords_preds': coords_preds,
            'existence_preds': existence_preds
        }
