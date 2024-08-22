import torch

DIR_DISCRETIZATON = 24
MAX_DIR = 2*torch.pi
MIN_DIR = 0

NORM_DISCRETIZATION = 16
MAX_NORM = 0.1 
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
    # normalize
    actions[:, :, :, :, 0] = ((torch.clip(actions[:, :, :, :, 0], min=MIN_DIR, max=MAX_DIR) - MIN_DIR) / (MAX_DIR - MIN_DIR))
    actions[:, :, :, :, 1] = ((torch.clip(actions[:, :, :, :, 1], min=MIN_NORM, max=MAX_NORM) - MIN_NORM) / (MAX_NORM - MIN_NORM))
    
    # discretize the actions
    actions[:, :, :, :, 0] = torch.round(actions[:, :, :, :, 0] * (DIR_DISCRETIZATON - 1))
    actions[:, :, :, :, 1] = torch.round(actions[:, :, :, :, 1] * (NORM_DISCRETIZATION - 1))

    # combine into a single categorical value
    actions = actions[:, :, :, :, 0] * NORM_DISCRETIZATION + actions[:, :, :, :, 1]

    return actions


def actions_to_bbox_seq(actions, initial_bboxes, frame_size):
    """
    actions: [batch_size=1, num_timesteps, num_agents, 2, 2]
    initial_bboxes: [batch_size=1, num_agents, 4]

    bboxes: [batch_size, num_timesteps, num_agents, 4]
    """

    batch_size, num_timesteps, num_agents, _, _ = actions.shape
    bboxes = torch.zeros([batch_size, num_timesteps, num_agents, 4], device=actions.device)
    bboxes[:, 0] = initial_bboxes
    
    for t in range(1, num_timesteps):
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


print("Vocabulary size:", VOCABULARY_SIZE)

frame_size = torch.tensor([320, 512])
bboxes = torch.tensor([[0, 0, 20, 20], [10, 10, 30, 30], [0, 0, 40, 40], [10, 0, 30, 40], [0, 10, 40, 30], [10, 10, 50, 50], [20, 20, 60, 60], [20, 20, 60, 60], [20, 20, 60, 70], [20, 20, 70, 70]])
bboxes = bboxes.unsqueeze(0).unsqueeze(2)

print(bboxes)

actions = bbox_seq_to_actions(bboxes, frame_size)

print(actions)

dis_actions = discretize_actions(actions)

print(dis_actions)

undis_actions = undiscretize_actions(dis_actions)

print(undis_actions)

bboxes_out = actions_to_bbox_seq(undis_actions, bboxes[:, 0], frame_size)

print(bboxes_out)

print("Original:", bboxes)
