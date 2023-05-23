import torch
import numpy as np


def reward_to_go_calculation(reward_decay_factor, trajectory_time, reward_record):
    reward_decay_tensor = torch.from_numpy(np.array([reward_decay_factor ** i for i in range(trajectory_time)])[None])
    # now perform reward cum_sum;
    reward_to_go = torch.zeros_like(reward_record)
    for i in range(trajectory_time - 1, -1, -1):
        reward_to_go[:, i] = torch.sum(reward_decay_tensor[:, :trajectory_time - i] * reward_record[:, i:], dim=1)
    return reward_to_go