import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import imageio

# ===================================== DISCRETIZATON =====================================
DIR_DISCRETIZATON = 24 #16
MAX_DIR = 2*torch.pi
MIN_DIR = 0

NORM_DISCRETIZATION = 16 #24
MAX_NORM = 0.1  # NOTE: Determined approximately by looking at data, may be other more reasonable values
MIN_NORM = 0.0

VOCABULARY_SIZE = DIR_DISCRETIZATON * NORM_DISCRETIZATION


def undiscretize_actions(actions):
    """
    input: actions [batch_size, num_timesteps, num_agents, 2]

    output: actions [batch_size, num_timesteps, num_agents, 2, 2]
    """
    # Initialize the array for the continuous actions
    actions_shape = (actions.shape[0], actions.shape[1], actions.shape[2], 2, 2)
    continuous_actions = torch.zeros(actions_shape, device=actions.device)
    
    # Separate the combined actions back into their discretized components
    continuous_actions[:, :, :, :, 0] = actions // NORM_DISCRETIZATION
    continuous_actions[:, :, :, :, 1] = actions % NORM_DISCRETIZATION
    
    # Reverse the discretization
    continuous_actions[:, :, :, :, 0] /= (DIR_DISCRETIZATON - 1)
    continuous_actions[:, :, :, :, 1] /= (NORM_DISCRETIZATION - 1)
    
    # Denormalize to get back the original continuous values
    continuous_actions[:, :, :, :, 0] = (continuous_actions[:, :, :, :, 0] * (MAX_DIR - MIN_DIR)) + MIN_DIR
    continuous_actions[:, :, :, :, 1] = (continuous_actions[:, :, :, :, 1] * (MAX_NORM - MIN_NORM)) + MIN_NORM
    
    return continuous_actions


def discretize_actions(actions):
    """
    actions: [batch_size, num_timesteps, num_agents, 2, 2]
    """

    actions_out = torch.zeros_like(actions)

    # normalize
    actions_out[:, :, :, :, 0] = ((torch.clip(actions[:, :, :, :, 0], min=MIN_DIR, max=MAX_DIR) - MIN_DIR) / (MAX_DIR - MIN_DIR))
    actions_out[:, :, :, :, 1] = ((torch.clip(actions[:, :, :, :, 1], min=MIN_NORM, max=MAX_NORM) - MIN_NORM) / (MAX_NORM - MIN_NORM))
    
    # discretize the actions
    actions_out[:, :, :, :, 0] = torch.round(actions_out[:, :, :, :, 0] * (DIR_DISCRETIZATON - 1))
    actions_out[:, :, :, :, 1] = torch.round(actions_out[:, :, :, :, 1] * (NORM_DISCRETIZATION - 1))

    # combine into a single categorical value
    actions_out = actions_out[:, :, :, :, 0] * NORM_DISCRETIZATION + actions_out[:, :, :, :, 1]

    return actions_out


def actions_to_bbox_seq(actions, initial_bboxes, frame_size, discard_first_action=False):
    """
    actions: [batch_size=1, num_timesteps, num_agents, 2, 2]
    initial_bboxes: [batch_size=1, num_agents, 4]

    bboxes: [batch_size, num_timesteps, num_agents, 4]
    """

    batch_size, num_timesteps, num_agents, _, _ = actions.shape
    bboxes = torch.zeros([batch_size, num_timesteps, num_agents, 4], device=actions.device)
    bboxes[:, 0] = initial_bboxes

    offset = 1 if discard_first_action else 0
    for t in range(offset, num_timesteps):
        direction1 = actions[:, t, :, 0, 0]
        direction2 = actions[:, t, :, 1, 0]
        
        norm1 = actions[:, t, :, 0, 1]
        norm2 = actions[:, t, :, 1, 1]

        # Previous bbox points
        x1 = bboxes[:, t-1, :, 0]
        y1 = bboxes[:, t-1, :, 1]
        x2 = bboxes[:, t-1, :, 2]
        y2 = bboxes[:, t-1, :, 3]
        
        dx1 = norm1 * torch.cos(direction1)
        dx2 = norm2 * torch.cos(direction2)
        dy1 = norm1 * torch.sin(direction1)
        dy2 = norm2 * torch.sin(direction2)
        
        x1p = x1 + dx1 * frame_size[0]
        x2p = x2 + dx2 * frame_size[0]
        y1p = y1 + dy1 * frame_size[1]
        y2p = y2 + dy2 * frame_size[1]

        next_bboxes = torch.stack([x1p, y1p, x2p, y2p], dim=-1)
        bboxes[:, t] = next_bboxes
    
    return bboxes


def bbox_seq_to_actions(bboxes, frame_size):
    """
    Convert bbox sequence to actions
    bboxes: [batch_size, timesteps, num_agents, 4]

    actions: [batch_size, timesteps, num_agents, 2, 2]
    """
    batch_size, timesteps, num_agents, _ = bboxes.shape
    actions = torch.zeros([batch_size, timesteps, num_agents, 2, 2], device=bboxes.device)
    norms1 = []
    norms2 = []
    for i in range(1, len(bboxes[0])):
        x1 = bboxes[:, i-1, :, 0]
        y1 = bboxes[:, i-1, :, 1]
        x2 = bboxes[:, i-1, :, 2]
        y2 = bboxes[:, i-1, :, 3]

        x1p = bboxes[:, i, :, 0]
        y1p = bboxes[:, i, :, 1]
        x2p = bboxes[:, i, :, 2]
        y2p = bboxes[:, i, :, 3]
        
        # Normalizing coordinates to a frame with 1:1 aspect ratio (square)
        dy1 = (y1p - y1) / frame_size[1]
        dy2 = (y2p - y2) / frame_size[1]
        dx1 = (x1p - x1) / frame_size[0]
        dx2 = (x2p - x2) / frame_size[0]

        # Compute angles and normalize from [-pi, pi] to [0, 2*pi]
        direction1 = torch.remainder(torch.arctan2(dy1, dx1) + 2*torch.pi, 2*torch.pi)
        direction2 = torch.remainder(torch.arctan2(dy2, dx2) + 2*torch.pi, 2*torch.pi)

        norm1 = torch.sqrt(dx1**2 + dy1**2) 
        norm2 = torch.sqrt(dx2**2 + dy2**2)

        norms1.append(norm1)
        norms2.append(norm2)

        p1_action = torch.concatenate([direction1.reshape(batch_size, num_agents, 1), norm1.reshape(batch_size, num_agents, 1)], axis=-1)
        p2_action = torch.concatenate([direction2.reshape(batch_size, num_agents, 1), norm2.reshape(batch_size, num_agents, 1)], axis=-1)
        actions[:, i, :, 0] = p1_action
        actions[:, i, :, 1] = p2_action
    
    return actions

# ===================================== DISCRETIZATON END =====================================

# ===================================== DATA PROCESSING =======================================

def reshape_data(tensor, track_ids):
    """
    Expecting tensor with 4 dimensions [batch, timesteps, num_agents, data]
    Aligns the data from tensor to be at the index corresponding to the id in track_ids
    """
    tensor_out = torch.zeros_like(tensor)

    # Create a mask for valid track_ids (track_id != -1)
    valid_mask = track_ids != -1

    # Get the indices where the track_ids are valid
    valid_indices = valid_mask.nonzero(as_tuple=True)

    # Extract the valid bbox coordinates and corresponding indices
    valid_bboxes = tensor[valid_indices[0], valid_indices[1], valid_indices[2]]
    valid_agent_ids = track_ids[valid_mask]

    # Scatter the valid bbox coordinates into the output tensor
    tensor_out[valid_indices[0], valid_indices[1], valid_agent_ids] = valid_bboxes

    return tensor_out


# NOTE: This could be done in preprocessing to speed up training
def normalize_track_ids(track_ids):
    """
    Recast to values in [0, num_agents[
    """
    batch_size, timesteps, max_num_agents = track_ids.shape

    # Disambiguate id=0 (located in first position) with 0 indicating null id, by setting null ids to -1
    first_pos_mask = torch.zeros_like(track_ids, dtype=torch.bool)
    first_pos_mask[:, :, 0] = track_ids[:, :, 0] == 0
    zero_mask = (track_ids == 0)
    null_id_mask = (zero_mask & ~first_pos_mask)
    track_ids[null_id_mask] = -1

    # Initialize an output tensor with zeros
    new_ids = torch.ones_like(track_ids) * -1

    # Process each batch separately
    for batch in range(batch_size):
        unique_ids = track_ids[batch].unique()
        unique_ids = unique_ids[unique_ids != -1]
        
        # Create a mapping from original ids to new indices
        id_to_new_index = {old_id.item(): new_index for new_index, old_id in enumerate(unique_ids, start=0)}
        
        for t in range(timesteps):
            for agent in range(max_num_agents):
                old_id = track_ids[batch, t, agent].item()
                if old_id != -1:
                    new_ids[batch, t, agent] = id_to_new_index[old_id]
    
    return new_ids


def process_data(object_data, out_frame_size=(512, 320), bbox_frame_size=(1382, 512)):
    # NOTE: This is currently for data from kitti

    # TODO: Potentially only consider bboxes above a certain confidence threshold from data
    bboxes = object_data['bbox']         # [batch_size, timesteps, num_agents, state_dim] (state_dim = 4 for {x1, y1, x2, y2})
    type_ids = object_data['id_type']    # [batch_size, timesteps, num_agents]
    track_ids = object_data['track_id']  # [batch_size, timesteps, num_agents]

    track_ids = normalize_track_ids(track_ids.clone())
    bboxes = reshape_data(bboxes, track_ids)
    type_ids = reshape_data(type_ids.unsqueeze(-1), track_ids)
    existence = bboxes[:, :, :, -1:].bool()  # Flag indicating if the agent's bbox exists at the current timestep (if it is in frame)

    # Rescale bbox coordinates
    bboxes[:, :, :, 0] *= out_frame_size[0] / bbox_frame_size[0]
    bboxes[:, :, :, 2] *= out_frame_size[0] / bbox_frame_size[0]
    bboxes[:, :, :, 1] *= out_frame_size[1] / bbox_frame_size[1]
    bboxes[:, :, :, 3] *= out_frame_size[1] / bbox_frame_size[1]
    
    # Convert bbox sequence to actions (action1: top left corner, action2: bottom right corner)
    actions = bbox_seq_to_actions(bboxes, bbox_frame_size)

    return {
        "actions": actions,
        "bboxes": bboxes,
        "type_ids": type_ids,
        "existence": existence
    }

# ===================================== DATA PROCESSING END =======================================

class MLPLayer(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(weight_init)

    def forward(self, x):
        return self.mlp(x)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
    

def create_lambda_lr(cfg):
    return lambda current_step: (
        current_step / cfg.lr_warmup_steps if current_step < cfg.lr_warmup_steps else max(0.0, (cfg.max_steps - current_step) / (cfg.max_steps - cfg.lr_warmup_steps))
    )


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, (nn.LSTM, nn.LSTMCell)):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, (nn.GRU, nn.GRUCell)):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)

def create_video_from_numpy_array(array, output_filename, fps=30):
    # Get the dimensions of the array
    num_frames, height, width, channels = array.shape

    # Create a video writer object
    with imageio.get_writer(output_filename, fps=fps) as writer:
        for i in range(num_frames):
            frame = array[i]
            # Append each frame to the video
            writer.append_data(frame)
    
    print(f'Video saved as {output_filename}')
