import argparse
import math
import random
import time
from collections import deque, namedtuple
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.nn.functional import mse_loss, cross_entropy
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from torch.distributions import Categorical


# run below lines for initializing python console.
def one_trajectory(env, trajectory_time, policy_net, value_net, device):
    trajectories = []
    rewards = []
    approximate_rewards = []
    log_probs = []
    prev_state = torch.from_numpy(env.reset()[0]).to(device)
    for i in range(trajectory_time):
        action, log_prob = policy_net.sample_discrete_action(prev_state)
        action_numpy = action.cpu().numpy()
        trajectories.append(prev_state.cpu().numpy())
        # realize log_prob is used for calculating gradient, cannot convert to numpy object!
        log_probs.append(log_prob[None])
        updated_state, reward, terminated, truncated, info = env.step(
            action_numpy)
        rewards.append(reward)
        prev_state = torch.from_numpy(updated_state).to(device)
        approximate_reward = value_net(prev_state)
        approximate_rewards.append(approximate_reward)
        if terminated:
            torch.from_numpy(env.reset()[0]).to(device)
            # print("terminated at time {}".format(i))
    log_probs_tensor = torch.cat(log_probs)
    approximate_rewards = torch.cat(approximate_rewards)
        # stack automatically adds a new dimension, and preserves gradient
    env.reset()  # always reset before performing next trajectory.
    return trajectories, rewards, approximate_rewards, log_probs_tensor


def reward_to_go_calculation(reward_decay_factor, trajectory_time, reward_record):
    reward_decay_tensor = torch.from_numpy(np.array([reward_decay_factor ** i
                                                     for i in range(trajectory_time)])[None])
    # now perform reward cum_sum;
    reward_to_go = torch.zeros_like(reward_record)
    for i in range(trajectory_time - 1, -1, -1):
        reward_to_go[:, i] = torch.sum(reward_decay_tensor[:, :trajectory_time - i]
                                       * reward_record[:, i:], dim=1)
    return reward_to_go


def train_networks(training_configs, device, env, policy_net, value_net, optimizer_value, optimizer_policy):
    epochs, batch_size, trajectory_time, reward_decay_factor = training_configs
    best_reward = -99999
    for training_epoch in range(epochs):
        log_probs_record, reward_record, approximate_reward_record, \
        states_record = one_epoch_data(batch_size, device, env,
                                                policy_net, value_net, trajectory_time)

        gt_reward_record = reward_to_go_calculation(reward_decay_factor,
                                                    trajectory_time, reward_record).to(device)
        # note: observations of gt_reward record shows, perhaps the results needs to be normalized: most of the time
        # the reward would be much higher than 1, due to accumulation. Would this lead to problems of the network?
        # loss for approximation models will be calculated as normalized values;

        value_net_loss = torch.mean(torch.linalg.norm(gt_reward_record - approximate_reward_record, dim=1))
        optimizer_value.zero_grad()
        value_net_loss.backward()
        optimizer_value.step()

        # now start fitting policy net
        # first calculate A value;
        a_value = value_net(states_record.to(device)).flatten(-2, -1) # use updated value_net to acquire new reward
                                    # for each sampled states
        a_value = reward_record.to(device) + torch.concatenate((a_value[:, 1:],
                                    torch.zeros([batch_size, 1]).to(device)), dim=1) - a_value
        policy_net_loss = torch.sum((-1) * (torch.cat(log_probs_record)) * a_value)
        optimizer_policy.zero_grad()
        policy_net_loss.backward()
        optimizer_policy.step()

        # training output
        mean_reward = torch.mean(reward_record)
        if mean_reward > best_reward:
            best_reward = mean_reward
        if mean_reward > 0 or training_epoch % 100 == 0:
            print("epoch {} with reward {}; "
                  "policy_net_loss: {}, "
                  "value_net_loss: {}".format(training_epoch, torch.mean(reward_record),
                                              policy_net_loss, value_net_loss))
    print(best_reward)


def one_epoch_data(batch_size, device, env, policy_net, value_net, trajectory_time):
    trajectory_record = []
    reward_record = []
    approximate_reward_record = []
    log_probs_record = []
    for batch in range(batch_size):
        trajectories, rewards, approximate_rewards, log_probs = one_trajectory(
            env, trajectory_time, policy_net, value_net, device)
        trajectory_record.append(trajectories)
        reward_record.append(rewards)
        approximate_reward_record.append(approximate_rewards)
        log_probs_record.append(log_probs[None])

    return log_probs_record, torch.Tensor(reward_record), \
           torch.stack(approximate_reward_record), torch.tensor(np.array(trajectory_record))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="LunarLander-v2")
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_size", type=int, default=25_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--list_layer", nargs="+", type=int, default=[64, 64])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_start", type=int, default=1)
    parser.add_argument("--eps_decay", type=int, default=50_000)
    parser.add_argument("--learning_start", type=int, default=10_000)
    parser.add_argument("--train_frequency", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--capture_video", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    args.device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    return args


class ValueNetwork(nn.Module):

    def __init__(self, observation_shape, ):
        super().__init__()
        self.observation_shape = observation_shape
        self.input_layer = nn.Linear(observation_shape, observation_shape * 2)
        self.output_layer = nn.Linear(observation_shape * 2, 1) # generate an approximated value for current state,
                            # where the value is a scalar

    def forward(self, state):
        """
        if the environment has discrete action space, will return the likelihood for all states. otherwise,
        will sigmoid the actions to adjust scale within action space's range.
        :param state:
        :return:
        """
        x = self.input_layer(state)
        return self.output_layer(x)


class PolicyNetwork(nn.Module):

    def __init__(self, observation_shape, action_shape):
        super().__init__()
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.input_layer = nn.Linear(observation_shape, observation_shape * 2)
        self.intermediate_layer = nn.Linear(observation_shape * 2, action_shape * 2)
        self.output_layer = nn.Linear(action_shape * 2, action_shape)

    def forward(self, state):
        """
        if the environment has discrete action space, will return the likelihood for all states. otherwise,
        will sigmoid the actions to adjust scale within action space's range.
        :param state:
        :return:
        """
        x = self.input_layer(state)

        #relu activation
        x = nn.ReLU()(x)

        x = self.intermediate_layer(x)

        x = nn.ReLU()(x)

        #actions
        actions = self.output_layer(x)
        return actions

    def sample_discrete_action(self, state):
        """
        only called when the model has discrete action space. In that case will use categorical
        distribution and return log-likelihood for sampling.
        :param state:
        :return:
        """
        # first use network to predict a likelihood for each action.
        action_prob = nn.Softmax()(self.forward(state))
        action_distribution = Categorical(action_prob)
        sampled_action = action_distribution.sample()
        log_probab = action_distribution.log_prob(
            sampled_action)  # for calculating gradient
        return sampled_action, log_probab

    def sample_continuous_action(self, state, floor, ceil):
        """
        :param state:
        :param floor: the action value bounded below, a Tensor having same shape as action_shape
        :param ceil: the action value bounded above, a Tensor having same shape as "floor"
        :return:
        """
        # first use a normalization method to convert output action to standard range, then
        # use ceil and floor to normalize the action values.
        standard_conversion = nn.Sigmoid(
        )  # might need to be replaced to other methods when required
        output_action = standard_conversion(self.forward(state)).to("cpu")
        normalized_actions = floor + output_action * (
                ceil - floor)  # realize Sigmoid ranges from 0 to 1;
        return normalized_actions  # perhaps also need to add modifications for likelihood or loss calculation.


def get_exploration_prob(args, step):
    # Linear decay of epsilon
    return args.eps_end + (args.eps_start - args.eps_end) * math.exp(
        -1.0 * step / args.eps_decay)

