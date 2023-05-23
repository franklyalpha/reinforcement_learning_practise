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


def train(dynamics_training_params, policy_training_params, target_net, policy_net, optimizer_policy,
          dynamics_net, best_dynamics_net, optimizer_dynamics):
    dynamics_epoch, batch_size, sample_times, rollout_times, \
        rollout_length, env, device = dynamics_training_params
    lowest_dynamic_loss = 99999
    for i in range(dynamics_epoch):
        env_sample_buffer = ReplayBuffer(batch_size)
        # first sample a set of data and store in a replay_buffer.
        sample_datapoints(target_net, sample_times, env_sample_buffer, env, device)
    
        # after that use sampled replay buffer data to update dynamics model, using all the data from
        env_buffer_num_batch = 1 + len(env_sample_buffer) // env_sample_buffer.batch_size()
        dynamics_net_loss_record = -99999
        for j in range(int(env_buffer_num_batch)):
            batched_init_state, batched_actions, batched_next_state, \
            batched_reward = env_sample_buffer.generate_batch_sample(device)
            predicted_states, predicted_reward = dynamics_net(batched_init_state, batched_actions[:, None])
            dynamics_net_loss = mse_loss(predicted_states, batched_next_state) + \
                                mse_loss(predicted_reward, batched_reward)

            optimizer_dynamics.zero_grad()
            dynamics_net_loss.backward()
            optimizer_dynamics.step()
            dynamics_net_loss_record = max(float(dynamics_net_loss.detach()), dynamics_net_loss_record)
        print("dynamics_net_loss: " + str(dynamics_net_loss_record))
        # use previous dynamics network if loss is higher;
        if dynamics_net_loss_record < lowest_dynamic_loss:
            lowest_dynamic_loss = dynamics_net_loss_record
            best_dynamics_net.load_state_dict(dynamics_net.state_dict())
            print("best dynamics state dict loaded!")
        # env_sample_buffer.empty_buffer()

        with torch.no_grad():
            for k in range(rollout_times):
                sampled_state_batch = env_sample_buffer.generate_single_sample(device)[0]
                sampled_buffer = dynamics_net.rollout_future(rollout_length,
                                                             torch.tensor(sampled_state_batch).to(device), target_net)
                env_sample_buffer.integrate_replay_buffers(sampled_buffer, batch_size)
        train_networks(policy_training_params,
                       device, env, policy_net, target_net, env_sample_buffer, optimizer_policy)
        env_sample_buffer.empty_buffer()


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
            target_net.load_state_dict(
                policy_net.state_dict())  # realizing the returned values are immutable; don't worry about co-update issues.
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
        # if mean_reward > 0 or training_epoch % 100 == 0:
        if mean_reward > 0:
            print("epoch {} with reward {} "
                  "policy_net_loss: {}".format(training_epoch, mean_reward,
                                               policy_net_loss))
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


class DynamicsNetwork(nn.Module):

    def __init__(self, observation_shape, action_embedding_shape, action_input_shape, predict_reward=True):
        super().__init__()
        self.observation_shape = observation_shape
        self.action_embedding_shape = action_embedding_shape
        self.predict_reward = int(predict_reward)

        self.action_embedding = nn.Linear(action_input_shape, action_embedding_shape)
        self.input_layer = nn.Linear(observation_shape + action_embedding_shape,
                                     observation_shape * 2 + action_embedding_shape * 2)
        self.state_output_layer = nn.Linear(observation_shape * 2 + action_embedding_shape * 2,
                                            observation_shape + self.predict_reward) # generate an approximated value for current state,
                            # where the value is a scalar

    def forward(self, state, action):
        """
        return the predicted state and reward
        :param state: the current given state; has size [B, observation_shape]
        :param action: the action executed; has size [B, action_input_shape]
        :return: tensor representing predicted states and optionally, reward; has size [B, observation_shape + 1]
                if includes reward, and reward can be acquired using [:, -1] indexing to extract.
        """
        if len(action.shape) == 1:
            # extend dimension of action
            action = action[:, None]
        embedded_action = self.action_embedding(action.float())
        concatenated_input = self.input_layer(torch.cat([state, embedded_action], dim=1))
        outputs = self.state_output_layer(concatenated_input)
        return outputs[:, :-1], outputs[:, -1] # observation states, and rewards

    @torch.no_grad()
    def rollout_future(self, rollout_len, initial_state, policy_network):
        """
        perform rollouts for future states using the dynamics model.
        will return a list of tuples, in the form (s, a, s', r), starting from initial state
        :param rollout_len:
        :param initial_state: has shape [observation_shape]
        :param policy_network:
        :return:
        """
        record = ReplayBuffer(1)
        state = initial_state
        for i in range(rollout_len):
            action = policy_network.sample_discrete_action(state)[0]
            predicted_state, predicted_reward = self.forward(state[None], action[None][None].float())
            record.add_sample((state.cpu().numpy(), action.cpu().numpy(),
                               predicted_state[0].cpu().numpy(), predicted_reward.cpu().detach().numpy()[0]))
            state = predicted_state[0]
        return record


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

class ReplayBuffer:

    def __init__(self, batch_size):
        super(ReplayBuffer, self).__init__()
        self.buffer = []
        self._batch_size = batch_size

    def __len__(self):
        return len(self.buffer)

    def batch_size(self):
        return self._batch_size

    def add_sample(self, input_buffer):
        """
        :param input_buffer: should be a tuple containing the following:
        input_buffer[0] gives st, input_buffer[1] gives at, input_buffer[2] gives st',
        input_buffer[3] gives reward.
        :return: None
        """
        self.buffer.append((input_buffer[0].flatten(), input_buffer[1],
                            input_buffer[2].flatten(), float(input_buffer[3])))

    def generate_single_sample(self, device):
        """
        generate a random sample from stored replay buffer.
        :return: a tuple of input_buffer. access those elements in the same way
        those are passed in for storage.
        """
        generator_limit = len(self.buffer) - 1
        generated_num = random.randint(0, generator_limit)  # realizing random.randint includes both ends!!!
        init_state, action, next_state, reward = self.buffer[generated_num]
        return torch.Tensor(init_state).to(device), torch.Tensor(action).to(device), \
            torch.Tensor(next_state).to(device), torch.Tensor(np.array([reward])).to(device)

    def generate_batch_sample(self, device):
        indexes = random.sample(range(len(self.buffer)), self._batch_size)
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

    def integrate_replay_buffers(self, replay_buffer, batch_size=None):
        self.buffer.extend(replay_buffer.buffer)
        if batch_size is not None:
            self._batch_size = batch_size
        else:
            self._batch_size = max(self._batch_size, replay_buffer.batch_size())

    def empty_buffer(self):
        self.buffer = []
