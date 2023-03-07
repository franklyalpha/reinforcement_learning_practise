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
        # TODO: check assumption that approximate_reward is a Torch Tensor with shape [1], and is in "device"
        approximate_reward = value_net(prev_state)
        approximate_rewards.append(approximate_reward)
        if terminated:
            torch.from_numpy(env.reset()[0]).to(device)
    log_probs_tensor = torch.cat(log_probs)
    approximate_rewards = torch.cat(approximate_rewards)
        # stack automatically adds a new dimension, and preserves gradient
    env.reset()  # always reset before performing next trajectory.
    return trajectories, rewards, approximate_rewards, log_probs_tensor


def reward_to_go_calculation(reward_decay_factor, trajectory_time, reward_record):
    reward_decay_tensor = torch.from_numpy(np.array([reward_decay_factor ** i for i in range(trajectory_time)])[None])
    # now perform reward cum_sum;
    reward_to_go = torch.zeros_like(reward_record)
    for i in range(trajectory_time - 1, -1, -1):
        reward_to_go[:, i] = torch.sum(reward_decay_tensor[:, :trajectory_time - i] * reward_record[:, i:], dim=1)
    return reward_to_go


def train_networks(training_configs, device, env, policy_net, value_net, optimizer_value, optimizer_policy):
    epochs, batch_size, trajectory_time, reward_decay_factor = training_configs
    for training_epoch in range(epochs):
        log_probs_record, reward_record, approximate_reward_record, \
        states_record = one_epoch_data(batch_size, device, env,
                                                policy_net, value_net, trajectory_time)

        gt_reward_record = reward_to_go_calculation(reward_decay_factor,
                                                    trajectory_time, reward_record).to(device)
        # note: observations of gt_reward record shows, perhaps the results needs to be normalized: most of the time
        # the reward would be much higher than 1, due to accumulation. Would this lead to problems of the network?
        # loss for approximation models will be calculated as normalized values;

        value_net_loss = torch.sum(torch.linalg.norm(gt_reward_record - approximate_reward_record, dim=1))
        optimizer_value.zero_grad()
        value_net_loss.backward()
        optimizer_value.step()

        # now start fitting policy net
        # first calculate A value;
        a_value = value_net(states_record.to(device)).flatten(-2, -1)
        a_value = reward_record.to(device) + torch.concatenate((a_value[:, 1:],
                                    torch.zeros([batch_size, 1]).to(device)), dim=1) - a_value
        policy_net_loss = torch.sum(torch.cat(log_probs_record) * a_value)
        optimizer_policy.zero_grad()
        policy_net_loss.backward()
        optimizer_policy.step()
        print("epoch {} with reward {}; "
              "policy_net_loss: {}, "
              "value_net_loss: {}".format(training_epoch, torch.mean(reward_record),
                                          policy_net_loss, value_net_loss))


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


# def make_env(env_id, capture_video=False):
#
#     def thunk():
#
#         if capture_video:
#             env = gym.make(env_id, render_mode="rgb_array")
#             env = gym.wrappers.RecordVideo(
#                 env=env,
#                 video_folder=f"{run_dir}/videos/",
#                 episode_trigger=lambda x: x,
#                 disable_logger=True,
#             )
#         else:
#             env = gym.make(env_id)
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         env = gym.wrappers.FlattenObservation(env)
#         env = gym.wrappers.NormalizeObservation(env)
#         env = gym.wrappers.TransformObservation(
#             env, lambda obs: np.clip(obs, -10, 10))
#         env = gym.wrappers.NormalizeReward(env)
#         env = gym.wrappers.TransformReward(
#             env, lambda reward: np.clip(reward, -10, 10))
#
#         return env
#
#     return thunk


class ValueNetwork(nn.Module):

    def __init__(self, observation_shape, ):
        super().__init__()
        self.observation_shape = observation_shape
        self.network = nn.Linear(observation_shape, 1) # generate an approximated value for current state,
                            # where the value is a scalar

    def forward(self, state):
        """
        if the environment has discrete action space, will return the likelihood for all states. otherwise,
        will sigmoid the actions to adjust scale within action space's range.
        :param state:
        :return:
        """
        return self.network(state)


class PolicyNetwork(nn.Module):

    def __init__(self, observation_shape, action_shape):
        super().__init__()
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.network = nn.Linear(observation_shape, action_shape)

    def forward(self, state):
        """
        if the environment has discrete action space, will return the likelihood for all states. otherwise,
        will sigmoid the actions to adjust scale within action space's range.
        :param state:
        :return:
        """
        return self.network(state)

    def sample_discrete_action(self, state):
        """
        only called when the model has discrete action space. In that case will use categorical
        distribution and return log-likelihood for sampling.
        :param state:
        :return:
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

# if __name__ == "__main__":
#     args = parse_args()
#
#     date = str(datetime.now().strftime("%d-%m_%H:%M"))
#     # These variables are specific to the repo "rl-gym-zoo"
#     # You should change them if you are just copy/paste the code
#     algo_name = Path(__file__).stem.split("_")[0].upper()
#     run_dir = Path(
#         Path(__file__).parent.resolve().parents[1], "runs"
#         # , f"{args.env_id}__{algo_name}__{date}"
#     )
#
#     # Initialize wandb if needed (https://wandb.ai/)
#
#     # Create tensorboard writer and save hyperparameters
#     writer = SummaryWriter(run_dir)
#     writer.add_text(
#         "hyperparameters",
#         "|param|value|\n|-|-|\n%s" %
#         ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
#     )
#
#     # Set seed for reproducibility
#     if args.seed > 0:
#         random.seed(args.seed)
#         np.random.seed(args.seed)
#         torch.manual_seed(args.seed)
#
#     # Create vectorized environment
#     env = gym.vector.SyncVectorEnv([make_env(args.env_id)])
#
#     # Metadata about the environment
#     obversation_shape = env.single_observation_space.shape
#     action_shape = env.single_action_space.n
#
#     # Create the networks and the optimizer
#     policy_net = PolicyNetwork(args, obversation_shape, action_shape)
#
#     start_time = time.process_time()
#
#     # Main loop
#     for global_step in tqdm(range(args.total_timesteps)):
#
#         with torch.no_grad():
#             # Exploration or intensification
#             exploration_prob = get_exploration_prob(args, global_step)
#
#             # Log exploration probability
#             writer.add_scalar("rollout/eps_threshold", exploration_prob,
#                               global_step)
#
#             if np.random.rand() < exploration_prob:
#                 # Exploration
#                 action = torch.randint(action_shape, (1, )).to(args.device)
#             else:
#                 # Intensification
#                 action = torch.argmax(policy_net(state), dim=1)
#
#         # Perform action
#         next_state, reward, terminated, truncated, infos = env.step(
#             action.cpu().numpy())
#
#         # Convert transition to torch tensors
#         next_state = torch.from_numpy(next_state).to(args.device).float()
#         reward = torch.from_numpy(reward).to(args.device).float()
#         flag = torch.from_numpy(np.logical_or(terminated, truncated)).to(
#             args.device).float()
#
#         # Store transition in the replay buffer
#         # replay_buffer.push(state, action, reward, next_state, flag)
#
#         state = next_state
#
#         # Log episodic return and length
#         if "final_info" in infos:
#             info = infos["final_info"][0]
#
#             log_episodic_returns.append(info["episode"]["r"])
#             writer.add_scalar("rollout/episodic_return", info["episode"]["r"],
#                               global_step)
#             writer.add_scalar("rollout/episodic_length", info["episode"]["l"],
#                               global_step)
#
#         # Perform training step
#         if global_step > args.learning_start:
#             if not global_step % args.train_frequency:
#                 # Sample a batch from the replay buffer
#                 states, actions, rewards, next_states, flags = replay_buffer.sample(
#                 )
#
#                 # Compute TD error
#                 td_predict = policy_net(states).gather(1, actions).squeeze()
#
#                 # Compute TD target
#                 with torch.no_grad():
#                     # Double Q-Learning
#                     action_by_qvalue = policy_net(next_states).argmax(
#                         1).unsqueeze(-1)
#                     max_q_target = target_net(next_states).gather(
#                         1, action_by_qvalue).squeeze()
#
#                 td_target = rewards + (1.0 - flags) * args.gamma * max_q_target
#
#                 # Compute loss
#                 loss = mse_loss(td_predict, td_target)
#
#                 # Update policy network
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#
#                 # Update target network (soft update)
#                 for param, target_param in zip(policy_net.parameters(),
#                                                target_net.parameters()):
#                     target_param.data.copy_(args.tau * param.data +
#                                             (1 - args.tau) * target_param.data)
#
#                 # Log training metrics
#                 writer.add_scalar("train/loss", loss, global_step)
#
#         writer.add_scalar(
#             "rollout/SPS",
#             int(global_step / (time.process_time() - start_time)), global_step)
#
#     # Average of episodic returns (for the last 5% of the training)
#     indexes = int(len(log_episodic_returns) * 0.05)
#     avg_final_rewards = np.mean(log_episodic_returns[-indexes:])
#     print(
#         f"Average of the last {indexes} episodic returns: {round(avg_final_rewards, 2)}"
#     )
#     writer.add_scalar("rollout/avg_final_rewards", avg_final_rewards,
#                       global_step)
#
#     # Close the environment
#     env.close()
#     writer.close()
#
#
#     # Capture video of the policy
#     if args.capture_video:
#         print(f"Capturing videos and saving them to {run_dir}/videos ...")
#         env_test = gym.vector.SyncVectorEnv(
#             [make_env(args.env_id, capture_video=True)])
#         state, _ = env_test.reset()
#         count_episodes = 0
#
#         while count_episodes < 10:
#             with torch.no_grad():
#                 state = torch.from_numpy(state).to(args.device).float()
#                 action = torch.argmax(policy_net(state), dim=1).cpu().numpy()
#
#             state, _, terminated, truncated, _ = env_test.step(action)
#
#             if terminated or truncated:
#                 count_episodes += 1
#
#         env_test.close()
#         print("Done!")
