import argparse
import math
import random
import numpy as np
import torch
from torch import nn, optim
from torch.nn.functional import mse_loss, cross_entropy
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from torch.distributions import Categorical


# run below lines for initializing python console.
def one_trajectory(env, trajectory_time, policy_net, device):
    """

    :param env:
    :param trajectory_time:
    :param policy_net:
    :param device:
    :return: the set of trajectories, in python list format;
        the list of reward, with shape [trajectory_time]
        tensor of log_probability, with shape [1, trajectory_time]
    """
    trajectories = []
    rewards = []
    log_probs = []
    prev_state = torch.from_numpy(env.reset()[0]).to(device)
    for i in range(trajectory_time):
        action, log_prob = policy_net.sample_discrete_action(prev_state)  # both outputs are scalar
        action_numpy = action.cpu().numpy()
        trajectories.append([prev_state.cpu().numpy(), action_numpy])
        # realize log_prob is used for calculating gradient, cannot convert to numpy object!
        log_probs.append(log_prob[None])
        updated_state, reward, terminated, truncated, info = env.step(
            action_numpy)
        rewards.append(reward)  # reward is just float in python, not in numpy or in torch type.
        prev_state = torch.from_numpy(updated_state).to(device)
        if terminated:
            # return this trajectory for analysis, but will ensure shape consistency with trajectory time
            remaining_time = trajectory_time - i - 1
            log_probs.extend([log_prob[None]] * remaining_time)
            rewards.extend([0.0] * remaining_time)
            # print(f"terminated at trajectory time {i}")
            break
    log_probs_tensor = torch.cat(log_probs)
    env.reset()  # always reset before performing next trajectory.
    return trajectories, rewards, log_probs_tensor


def reward_to_go_calculation(reward_decay_factor, trajectory_time, reward_record):
    """

    :param reward_decay_factor:
    :param trajectory_time:
    :param reward_record: tensor of shape [batch_size, trajectory_time], containing reward for
        each single time step.
    :return: cumulative reward at each time step, with shape consistent with reward_record;
        each value at [:, t] would be the summation of reward from "t" to "trajectory_time"
    """
    reward_decay_tensor = torch.from_numpy(np.array([reward_decay_factor ** i for i in range(trajectory_time)])[None])
    # now perform reward cum_sum;
    reward_to_go = torch.zeros_like(reward_record)
    for i in range(trajectory_time - 1, -1, -1):
        reward_to_go[:, i] = torch.sum(reward_decay_tensor[:, :trajectory_time - i] * reward_record[:, i:], dim=1)
    return reward_to_go


def train_policy_network(training_configs, device, env, policy_net, optimizer):
    """
    Basic training mechanism for policy gradient method
    :param training_configs:
    :param device:
    :param env:
    :param policy_net:
    :param optimizer:
    :return:
    """
    epochs, batch_size, trajectory_time, reward_decay_factor = training_configs
    best_reward = -99999
    for training_epoch in range(epochs):
        trajectory_record = []
        reward_record = []
        log_probs_record = []
        for batch in range(batch_size):
            trajectories, rewards, log_probs = one_trajectory(env, trajectory_time, policy_net, device)
            trajectory_record.append(trajectories)
            reward_record.append(rewards)
            log_probs_record.append(log_probs[None])

        log_probs_record = torch.cat(log_probs_record, dim=0)
        reward_record = torch.from_numpy(np.array(reward_record))
        # implement reward to go, with delayed rewards
        reward_to_go = reward_to_go_calculation(reward_decay_factor, trajectory_time, reward_record)
        reward_baseline = reward_to_go - torch.mean(reward_to_go, dim=1, keepdim=True)  # for training stabilization
        # del trajectories, rewards, log_probs
        # now start calculating gradient
        gradient = torch.sum(-log_probs_record * reward_baseline.to(device)) / batch_size
        # never forget the negative sign!!!
        optimizer.zero_grad()
        gradient.backward()
        optimizer.step()

        # calculate average reward
        mean_reward = torch.mean(reward_record)
        if mean_reward > best_reward:
            best_reward = mean_reward
            print("epoch {} with mean reward {}".format(training_epoch, mean_reward))
        if training_epoch % 25 == 0:
            print("epoch {} with mean reward {}".format(training_epoch, mean_reward))
    print(best_reward)


class PolicyNetwork(nn.Module):

    def __init__(self, observation_shape, action_shape):
        super().__init__()
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.input_layer = nn.Linear(observation_shape, observation_shape * 2)
        self.intermediate_layer = nn.Linear(observation_shape * 2, action_shape * 2)
        self.output_layer = nn.Linear(action_shape * 2, action_shape)
        self.network = nn.Sequential(self.input_layer, nn.ReLU(),
                                     self.intermediate_layer, nn.ReLU(),
                                     self.output_layer)

    def forward(self, state):
        """
        if the environment has discrete action space, will return the likelihood for all states. otherwise,
        will sigmoid the actions to adjust scale within action space's range.
        :param state: has shape [..., self.observation_shape]
        :return: logits, with shape [..., self.action_shape]
        """
        return self.network(state)

    def sample_discrete_action(self, state):
        """
        only called when the model has discrete action space. In that case will use categorical
        distribution and return log-likelihood for sampling.
        :param state: tensor with shape [..., self.observation_shape]
        :return: sampled_action, log_probab, both are scalar in tensor type
        """
        # first use network to predict a likelihood for each action.
        action_prob = nn.Softmax()(self.network(state))
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
        output_action = standard_conversion(self.network(state)).to("cpu")
        normalized_actions = floor + output_action * (
                ceil - floor)  # realize Sigmoid ranges from 0 to 1;
        return normalized_actions  # perhaps also need to add modifications for likelihood or loss calculation.


def get_exploration_prob(args, step):
    # Linear decay of epsilon
    return args.eps_end + (args.eps_start - args.eps_end) * math.exp(
        -1.0 * step / args.eps_decay)
