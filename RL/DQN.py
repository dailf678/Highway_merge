from typing import Optional
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from torch import FloatTensor, LongTensor, ByteTensor
from collections import namedtuple
import random
from torch.autograd import Variable
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from utils import get_linear_fn, get_schedule_fn
from stable_baselines3.common.logger import Logger
from stable_baselines3.common import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNNet, self).__init__()
        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 256)
        self.linear4 = nn.Linear(256, action_dim)

    def forward(self, s):
        s = F.relu(self.linear1(s))
        s = F.relu(self.linear2(s))
        s = F.relu(self.linear3(s))
        s = self.linear4(s)
        return s


class DQN(object):
    def __init__(
        self,
        env,
        lr,
        gamma,
        epsilon,
        MEMORY_CAPACITY,
        state_dim,
        action_dim,
        learning_starts: int = 200,
        targe_update_f: int = 50,
        batch_size: int = 32,
        dqn_type: str = "DQN",
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        exploration_fraction: float = 0.1,
        verbose: int = 0,
        tensorboard_log: Optional[str] = None,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
    ):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.learn_step_counter = 0
        self.targe_update_f = targe_update_f
        self.position = 0
        self.memory = []
        self.capacity = MEMORY_CAPACITY
        self.net = DQNNet(state_dim, action_dim)
        self.target_net = DQNNet(state_dim, action_dim)
        # self.loss_func = nn.MSELoss()
        self.loss_func = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.num_timesteps = 0
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.episode_rewards = 0
        self.reward_list = []
        self.dqn_type = dqn_type
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self._current_progress_remaining = 1
        self.exploration_rate = 0.0
        self.verbose = verbose
        self.tensorboard_log = tensorboard_log
        self.logger = utils.configure_logger(
            self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps
        )
        self.episode_num = 0

    def choose_action(self, s):
        x = torch.FloatTensor(s)
        if np.random.uniform() < 1 - self.exploration_rate:
            action = self.net.forward(x)
            action = torch.max(action, -1)[1].data.numpy()
            action = action
        else:
            action = np.random.randint(0, 5)
        return action

    def _push_memory(self, s, a, r, s_, done, truncated):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(
            torch.unsqueeze(torch.FloatTensor(s), 0),
            torch.unsqueeze(torch.FloatTensor(s_), 0),
            torch.from_numpy(np.array([a])),
            torch.from_numpy(np.array([r], dtype="float32")),
            torch.from_numpy(np.array([done], dtype="float32")),
        )  #
        self.position = (self.position + 1) % self.capacity
        if done or truncated:
            self.reset = self.env.reset()
            self._last_obs = self.reset[0]
            self._last_obs = self._last_obs.flatten()
        else:
            self._last_obs = s_

    def get_sample(self, batch_size):
        sample = random.sample(self.memory, batch_size)
        return sample

    def _setup_learn(self):
        self.reset = self.env.reset()
        self._last_obs = self.reset[0]
        self._last_obs = self._last_obs.flatten()
        self._current_progress_remaining = self.exploration_initial_eps
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )
        self.lr_schedule = get_schedule_fn(self.lr)

    def _on_step(self):
        self.exploration_rate = self.exploration_schedule(
            self._current_progress_remaining
        )
        # self.logger.record("rollout/exploration_rate", self.exploration_rate)
        # self.logger.record("time/exploration_rate", self.exploration_rate * 100)
        # self.logger.dump(step=self.num_timesteps)
        # print(self.exploration_rate)

        # for param_group in self.optimizer.param_groups:
        #     param_group["lr"] = self.lr_schedule(self._current_progress_remaining)
        print(self.num_timesteps)

    def collect_rollouts(self, env):
        self.num_timesteps += 1
        action = self.choose_action(self._last_obs)
        new_obs, reward, done, truncated, info = env.step(action)
        new_obs = new_obs.flatten()
        self._push_memory(self._last_obs, action, reward, new_obs, done, truncated)
        # self.env.render()
        self.episode_rewards += reward
        if done or truncated:
            self.reward_list.append(self.episode_rewards)
            self.episode_rewards = 0
            self.episode_num += 1
        # if self.episode_num % 4 == 0:
        #     self.logger.record("rollout/exploration_rate", self.exploration_rate)
        #     self.logger.record("time/exploration_rate", self.exploration_rate * 100)
        #     self.logger.dump(step=self.num_timesteps - 1)

    def train(self):
        self.learn_step_counter += 1
        if self.learn_step_counter % self.targe_update_f == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        transitions = self.get_sample(batch_size=self.batch_size)
        batch = Transition(*zip(*transitions))

        b_s = Variable(torch.cat(batch.state))
        b_s_ = Variable(torch.cat(batch.next_state))
        b_a = Variable(torch.cat(batch.action))
        b_r = Variable(torch.cat(batch.reward))
        b_d = Variable(torch.cat(batch.done))
        # 取self.net.forward(b_s).squeeze(1)，列的b_a.unsqueeze(1).to(torch.int64)
        q_eval = (
            self.net.forward(b_s).squeeze(1).gather(1, b_a.unsqueeze(1).to(torch.int64))
        )
        # q_eval = self.net.forward(b_s).gather(1, b_a.to(torch.int64))
        # 下个状态的最大Q值
        if self.dqn_type == "DDQN":  # DQN与Double DQN的区别
            max_action = self.net(b_s_).max(1)[1].view(self.batch_size, 1)
            max_next_q_values = self.target_net(b_s_).gather(1, max_action).detach()
        if self.dqn_type == "DQN":  # DQN的情况
            max_next_q_values = self.target_net(b_s_).detach()
            max_next_q_values = (
                self.target_net(b_s_).squeeze(1).max(1)[0].view(self.batch_size, 1)
            )
        # q_next = self.target_net.forward(b_s_).detach()
        # q_target = b_r.unsqueeze(1) + (
        #     1 - b_d.unsqueeze(1)
        # ) * self.gamma * q_next.squeeze(1).max(1)[0].view(self.batch_size, 1)
        q_target = (
            b_r.unsqueeze(1) + (1 - b_d.unsqueeze(1)) * self.gamma * max_next_q_values
        )
        loss = self.loss_func(q_eval, q_target)
        # print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def learn(self, total_timesteps: int):
        self._setup_learn()
        while self.num_timesteps < total_timesteps:
            self._on_step()
            self.collect_rollouts(self.env)
            if self.num_timesteps > self.learning_starts:
                self.train()
            self._current_progress_remaining = 1.0 - float(self.num_timesteps) / float(
                total_timesteps
            )
        return self

    def save(self, save_path):
        torch.save(self.target_net.state_dict(), save_path)
        # save self.reward_list
        np.save(
            "D:\python\Highway_merge\save_models\\return_list.npy", self.reward_list
        )

    def load(self, load_path):
        self.net.load_state_dict(torch.load(load_path))
        print("load model success")


Transition = namedtuple(
    "Transition", ("state", "next_state", "action", "reward", "done")
)
