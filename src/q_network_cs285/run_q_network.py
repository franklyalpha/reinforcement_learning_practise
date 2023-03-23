import numpy as np
import gymnasium as gym
import torch
from torch import nn, optim
from torch.nn.functional import mse_loss, cross_entropy
from tqdm import tqdm
import random
from torch.distributions import Categorical, Normal
import reinforcement_learning_practise.src.q_network_cs285.q_network as run_file

env = gym.make("LunarLander-v2")  # discrete action space
# env = gym.make('LunarLanderContinuous-v2') # this environment has both states and actions being continuous
state_shape = env.observation_space.shape[0]
action_shape = env.action_space.n  # for discrete action space, and discrete observation space
# action_shape = env.action_space.shape[0]
# action_bound_high = env.action_space.high
# action_bound_low = env.action_space.low

device = "cuda" if torch.cuda.is_available() else "cpu"
# reset environment

policy_net = run_file.PolicyNetwork(state_shape, action_shape)
target_net = run_file.PolicyNetwork(state_shape, action_shape)
policy_net.to(device)
target_net.to(device)
optimizer_policy = optim.Adam(policy_net.parameters(), lr=0.001)
# target_net does not require updates.

epochs, update_times, update_interval, gather_data, batch_size = 200, 20, 5, 50, 10

replay_buffer = run_file.ReplayBuffer(batch_size)

reward_decay_factor = 0.99


run_file.train_networks((epochs, update_times, update_interval, gather_data, batch_size, reward_decay_factor),
                              device, env, policy_net, target_net, replay_buffer, optimizer_policy)




