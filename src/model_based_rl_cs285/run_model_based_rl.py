import gymnasium as gym
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, cross_entropy
from torch import optim
import reinforcement_learning_practise.src.model_based_rl_cs285.model_based as run_file

env = gym.make("LunarLander-v2")  # discrete action space
# env = gym.make('LunarLanderContinuous-v2') # this environment has both states and actions being continuous
state_shape = env.observation_space.shape[0]
action_shape = env.action_space.n  # for discrete action space, and discrete observation space
# action_shape = env.action_space.shape[0]
# action_bound_high = env.action_space.high
# action_bound_low = env.action_space.low

device = "cuda" if torch.cuda.is_available() else "cpu"

policy_net = run_file.PolicyNetwork(state_shape, action_shape)
target_net = run_file.PolicyNetwork(state_shape, action_shape)
dynamics_net = run_file.DynamicsNetwork(state_shape, action_shape, action_input_shape=1)
policy_net.to(device)
target_net.to(device)
dynamics_net.to(device)
optimizer_policy = optim.Adam(policy_net.parameters(), lr=0.001)
optimizer_dynamics = optim.Adam(dynamics_net.parameters(), lr=0.001)

# configurations for training dynamics network
dynamics_epoch, batch_size, sample_times, rollout_times, rollout_length = 20, 10, 100, 20, 10  # realizing batch size of 5 is already terrible enough.
reward_decay_factor = 0.99
# configurations for training policy network
epochs, update_times, update_interval, gather_data = 20, 20, 5, 50

# Will implement Q-learning mechanism, with model-based data.
best_dynamics_net = run_file.DynamicsNetwork(state_shape,
                                             action_shape,
                                             action_input_shape=1)
best_dynamics_net.load_state_dict(dynamics_net.state_dict())

policy_training_params = (epochs, update_times, update_interval, gather_data, batch_size, reward_decay_factor)
dynamics_training_params = (dynamics_epoch, batch_size, sample_times, rollout_times, rollout_length,
                            env, device)

run_file.train(dynamics_training_params, policy_training_params, target_net, policy_net,
               optimizer_policy, dynamics_net, best_dynamics_net, optimizer_dynamics)
