import gymnasium as gym
import torch
from torch import optim
import reinforcement_learning_practise.src.actor_critic_cs285.actor_critic as run_file

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
value_net = run_file.ValueNetwork(state_shape)
policy_net.to(device)
value_net.to(device)
optimizer_policy = optim.Adam(policy_net.parameters(), lr=0.001)
optimizer_value = optim.Adam(value_net.parameters(), lr=0.001)

epochs, batch_size, trajectory_time = 10000, 1, 80 # realizing batch size of 5 is already terrible enough.
                                        # sampling one trajectory perhaps is always more preferred,
                                        # according to training results.
                                        # perhaps next time should only consider sampling only one trajectory,
                                        # and perform immediate updates.
reward_decay_factor = 0.99


run_file.train_networks((epochs, batch_size, trajectory_time, reward_decay_factor),
                              device, env, policy_net, value_net, optimizer_value, optimizer_policy)