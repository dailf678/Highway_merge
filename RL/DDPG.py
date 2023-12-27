import random
from collections import deque
import torch
import torch.nn as nn
import numpy as np

import torch.optim as optim
from noise import OUNoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate=1e-4):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = torch.relu(self.l1(state))
        x = torch.relu(self.l2(x))
        # 保证输出在(0,1)
        # 保证输出在（0.25，0.4）
        x = torch.sigmoid(self.l3(x)) * 0.15 + 0.25
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate=1e-4):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state, action):
        x = torch.relu(self.l1(state))
        x = torch.cat([x, action], 1)
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        return x


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class DDPG:
    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        buffer_size,
        batch_size,
        actor_lr,
        critic_lr,
        tau,
        gamme,
    ):
        self.actor = Actor(state_dim, hidden_dim, action_dim, actor_lr).to(device)
        self.actor_target = Actor(state_dim, hidden_dim, action_dim, actor_lr).to(
            device
        )
        self.critic = Critic(state_dim, hidden_dim, action_dim, critic_lr).to(device)
        self.critic_target = Critic(state_dim, hidden_dim, action_dim, critic_lr).to(
            device
        )

        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.batch_size = batch_size
        self.tau = tau
        self.gamme = gamme
        self.noise = OUNoise(n_action=action_dim, a_low=0.2, a_high=0.4)
        self._update_target_networks(tau=1)

    def choose_action(self, state, step):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action = self.actor(state).detach().numpy()[0]
        # return np.clip(action + noise, 0.2, 0.4)
        action = self.noise.get_action(action, step)
        return action

    def store_tansition(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample()

        states = np.array(states)
        states = torch.tensor(states, dtype=torch.float32).to(device)

        actions = np.array(actions)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)

        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)

        next_states = np.array(next_states)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)

        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        # Update critic
        self.critic.optimizer.zero_grad()

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q_values = self.critic_target(next_states, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamme * target_q_values

        current_q_values = self.critic(states, actions)

        critic_loss = nn.MSELoss()(current_q_values, target_q_values)
        critic_loss.backward()
        self.critic.optimizer.step()

        # Update Actor
        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic(states, self.actor(states)).mean()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Update target networks
        self._update_target_networks()

    def _update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau
        for target_param, param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
