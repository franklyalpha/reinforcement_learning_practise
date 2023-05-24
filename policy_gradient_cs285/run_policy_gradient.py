import gymnasium as gym
import torch
from torch import optim
import reinforcement_learning_practise.policy_gradient_cs285.policy_gradient as run_file


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
policy_net.to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)

epochs, batch_size, trajectory_time = 200 + 1, 5, 200  # just for debug. need to modify to other values when simple running works.
reward_decay_factor = 0.99

run_file.train_policy_network((epochs, batch_size, trajectory_time, reward_decay_factor),
                              device, env, policy_net, optimizer)



