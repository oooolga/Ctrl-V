import numpy as np
np.set_printoptions(suppress=True)

DIR_DISCRETIZATON = 24
MAX_DIR = 2 * np.pi
MIN_DIR = 0.0

NORM_DISCRETIZATION = 10
MAX_NORM = 0.05
MIN_NORM = 0.0

WIDTH_FACTOR_DISCRETIZATION = 10
MAX_WIDTH_FACTOR = 1.2
MIN_WIDTH_FACTOR = 0.8

HEIGHT_FACTOR_DISCRETIZATION = 10
MAX_HEIGHT_FACTOR = 1.2
MIN_HEIGHT_FACTOR = 0.8

VOCABULARY_SIZE = DIR_DISCRETIZATON * NORM_DISCRETIZATION * WIDTH_FACTOR_DISCRETIZATION * HEIGHT_FACTOR_DISCRETIZATION


def undiscretize_actions(actions):
    # Initialize the array for the continuous actions
    actions_shape = (actions.shape[0], actions.shape[1], 4)
    continuous_actions = np.zeros(actions_shape)
    
    # Separate the combined actions back into their discretized components
    ranges = [DIR_DISCRETIZATON, NORM_DISCRETIZATION, WIDTH_FACTOR_DISCRETIZATION, HEIGHT_FACTOR_DISCRETIZATION]
    continuous_actions[:, :, 0] = actions // np.prod(ranges[1:])
    remainder = actions % np.prod(ranges[1:])

    continuous_actions[:, :, 1] = remainder // np.prod(ranges[2:])
    remainder = remainder % np.prod(ranges[2:])

    continuous_actions[:, :, 2] = remainder // np.prod(ranges[3:])
    remainder = remainder % np.prod(ranges[3:])

    continuous_actions[:, :, 3] = remainder

    # print(continuous_actions)
    
    # Reverse the discretization
    continuous_actions[:, :, 0] /= (DIR_DISCRETIZATON - 1)
    continuous_actions[:, :, 1] /= (NORM_DISCRETIZATION - 1)
    continuous_actions[:, :, 2] /= (WIDTH_FACTOR_DISCRETIZATION - 1)
    continuous_actions[:, :, 3] /= (HEIGHT_FACTOR_DISCRETIZATION - 1)

    # print(continuous_actions)
    
    # Denormalize to get back the original continuous values
    continuous_actions[:, :, 0] = (continuous_actions[:, :, 0] * (MAX_DIR - MIN_DIR)) + MIN_DIR
    continuous_actions[:, :, 1] = (continuous_actions[:, :, 1] * (MAX_NORM - MIN_NORM)) + MIN_NORM
    continuous_actions[:, :, 2] = (continuous_actions[:, :, 2] * (MAX_WIDTH_FACTOR - MIN_WIDTH_FACTOR)) + MIN_WIDTH_FACTOR
    continuous_actions[:, :, 3] = (continuous_actions[:, :, 3] * (MAX_HEIGHT_FACTOR - MIN_HEIGHT_FACTOR)) + MIN_HEIGHT_FACTOR

    # print(continuous_actions)
    
    return continuous_actions


def discretize_actions(actions):
    # normalize
    actions[:, :, 0] = ((np.clip(actions[:, :, 0], a_min=MIN_DIR, a_max=MAX_DIR) - MIN_DIR) / (MAX_DIR - MIN_DIR))
    actions[:, :, 1] = ((np.clip(actions[:, :, 1], a_min=MIN_NORM, a_max=MAX_NORM) - MIN_NORM) / (MAX_NORM - MIN_NORM))
    actions[:, :, 2] = ((np.clip(actions[:, :, 2], a_min=MIN_WIDTH_FACTOR, a_max=MAX_WIDTH_FACTOR) - MIN_WIDTH_FACTOR) / (MAX_WIDTH_FACTOR - MIN_WIDTH_FACTOR))
    actions[:, :, 3] = ((np.clip(actions[:, :, 3], a_min=MIN_HEIGHT_FACTOR, a_max=MAX_HEIGHT_FACTOR) - MIN_HEIGHT_FACTOR) / (MAX_HEIGHT_FACTOR - MIN_HEIGHT_FACTOR))

    # print(actions)
    
    # discretize the actions
    actions[:, :, 0] = np.round(actions[:, :, 0] * (DIR_DISCRETIZATON - 1))
    actions[:, :, 1] = np.round(actions[:, :, 1] * (NORM_DISCRETIZATION - 1))
    actions[:, :, 2] = np.round(actions[:, :, 2] * (WIDTH_FACTOR_DISCRETIZATION - 1))
    actions[:, :, 3] = np.round(actions[:, :, 3] * (HEIGHT_FACTOR_DISCRETIZATION - 1))

    # print(actions)

    # combine into a single categorical value
    ranges = [DIR_DISCRETIZATON, NORM_DISCRETIZATION, WIDTH_FACTOR_DISCRETIZATION, HEIGHT_FACTOR_DISCRETIZATION]
    actions = actions[:, :, 0] * np.prod(ranges[1:]) + actions[:, :, 1] * np.prod(ranges[2:]) + actions[:, :, 2] * np.prod(ranges[3:]) + actions[:, :, 3]

    # print(actions)

    return actions

actions = np.random.rand(1, 10, 4)
actions[:, :, 0] = actions[:, :, 0] * 2*np.pi
actions[:, :, 1] = actions[:, :, 1] * 0.05
actions[:, :, 2] = actions[:, :, 2] * 0.4 + 0.8
actions[:, :, 3] = actions[:, :, 3] * 0.4 + 0.8

print(actions)

dis_actions = discretize_actions(actions)

print(dis_actions)

undis_actions = undiscretize_actions(dis_actions)

print(undis_actions)
