import argparse
import math
import random


import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.nn.functional import mse_loss, cross_entropy
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from torch.distributions import Categorical, Normal


def sample_datapoints(target_net, sample_data_count, replay_buffer, env, device):
    states, _ = env.reset()
    for i in range(sample_data_count):
        states = torch.tensor(states, device=device)  # this is to convert states into tensors
        new_action = np.array(target_net.sample_discrete_action(states)[0].cpu())
        updated_state, reward, terminated, truncated, info = env.step(
            new_action)
        if terminated:
            states, _ = env.reset()
            # do not record current sample.
            continue
        replay_buffer.add_sample((np.array(states.cpu()), new_action.copy(), updated_state.copy(), reward))
        states = updated_state


def train_networks(training_configs, device, env, policy_net, target_net, replay_buffer, optimizer_policy):
    epochs, update_times, update_interval, gather_data, batch_size, decay_factor = training_configs
    best_reward = -99999
    for training_epoch in range(epochs):
        # perform data_gathering; need to use policy network for performing future gradient updates.
        sample_datapoints(target_net, gather_data * batch_size, replay_buffer, env, device)
        # now start looped training, performing batch_size many updates on target_net for reference;
        # in each update the model network: policy_net, will be updated for trajectory_time many steps.
        reward_record = 0
        policy_net_loss = 0
        for j in range(update_times):
            target_net.load_state_dict(policy_net.state_dict())
            # realizing the returned values are immutable; don't worry about co-update issues.
            for k in range(update_interval):
                # calculate loss, by randomly picking up a sample from buffer.
                s, a, sp, r = replay_buffer.generate_batch_sample(device)
                q_value_sa = policy_net.q_value_calculation(s, a)  # contains gradient;
                # need to make sure target net doesn't contain gradient in the future.
                with torch.no_grad():
                    y_value = r.unsqueeze(-1) + target_net.sample_discrete_action(sp)[1].unsqueeze(-1) * decay_factor

                loss = mse_loss(q_value_sa, y_value)
                optimizer_policy.zero_grad()
                loss.backward()
                optimizer_policy.step()

                reward_record += torch.mean(r)
                policy_net_loss += loss
        replay_buffer.empty_buffer()
        # training output
        mean_reward = reward_record / (update_interval * update_times)
        policy_net_loss /= (update_interval * update_times)
        if mean_reward > best_reward:
            best_reward = mean_reward
        if mean_reward > 0 or training_epoch % 100 == 0:
            print("epoch {} with reward {} "
                  "policy_net_loss: {}".format(training_epoch, mean_reward,
                                               policy_net_loss))
    print(best_reward)


class ReplayBuffer:
    def __init__(self, batch_size):
        super(ReplayBuffer, self).__init__()
        self.buffer = []
        self.batch_size = batch_size

    def add_sample(self, input_buffer):
        """
        :param input_buffer: should be a tuple containing the following:
        input_buffer[0] gives st, input_buffer[1] gives at, input_buffer[2] gives st',
        input_buffer[3] gives reward.
        :return: None
        """
        self.buffer.append(input_buffer)

    def generate_single_sample(self):
        """
        generate a random sample from stored replay buffer.
        :return: a tuple of input_buffer. access those elements in the same way
        those are passed in for storage.
        """
        generator_limit = len(self.buffer) - 1
        generated_num = random.randint(0, generator_limit)  # realizing random.randint includes both ends!!!
        return self.buffer[generated_num]

    def generate_batch_sample(self, device):
        indexes = random.sample(range(len(self.buffer)), self.batch_size)
        initial_states = []
        actions = []
        next_states = []
        rewards = []
        for index in indexes:
            initial_s, action, next_s, reward = self.buffer[index]
            initial_states.append(initial_s)
            actions.append(int(action))
            next_states.append(next_s)
            rewards.append(reward)
        initial_states = torch.Tensor(np.array(initial_states))
        actions = torch.Tensor(actions)
        next_states = torch.Tensor(np.array(next_states))
        rewards = torch.Tensor(rewards)
        return initial_states.to(device), actions.to(device, dtype=torch.int64), next_states.to(device), rewards.to(
            device)

    def empty_buffer(self):
        self.buffer = []


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

        # relu activation
        x = nn.ReLU()(x)

        x = self.intermediate_layer(x)

        x = nn.ReLU()(x)

        # actions
        actions = self.output_layer(x)
        return actions
        # realizing that Q learning will provide values as return values, for policy to choose the
        # action (argmax) corresponding to highest outcome value.

    def sample_discrete_action(self, state):
        """
        for DQN, the value is being predicted by policy network. Thus usually the action is
        given by the index that yields the maximum reward. At the same time, the predicted reward will also
        be returned.

        could consider implementing exploration-exploitation trade-off in the future.

        :param state:
        :return:
        """
        predicted_value = self.forward(state)
        optimal_action = torch.argmax(predicted_value, dim=-1)
        max_value = torch.max(predicted_value, dim=-1)
        return optimal_action, max_value[0]

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

    def q_value_calculation(self, states, actions):
        """
        calculate the value for iterative updates for a fixed
        :param state:
        :param action:
        :return:
        """
        # np.choose is similar to torch.gather
        predicted_values = self.forward(states)
        q_value = torch.gather(predicted_values, -1, actions.unsqueeze(-1))  # have size [batch_size, 1]
        return q_value


def get_exploration_prob(args, step):
    # Linear decay of epsilon
    return args.eps_end + (args.eps_start - args.eps_end) * math.exp(
        -1.0 * step / args.eps_decay)
