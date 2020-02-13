
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
from collections import namedtuple


class Net(nn.Module):
    """Simple neural network for Deep-Q-Learning
    """
    def __init__(self, in_size, out_size, layers=[]):
        super(Net, self).__init__()
        self.layers = nn.ModuleList([])
        for size in layers:
            self.layers.append(nn.Linear(in_size, size))
            in_size = size
        self.layers.append(nn.Linear(in_size, out_size))

    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = F.leaky_relu(x)
            x = self.layers[i](x)
        return x


# Transition object
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ReplayMemory(object):
    """Collects transitions and samples minibatches."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DeepQlearningAgent():

    def __init__(self, state_dim, n_actions, replay_memory_capacity=100000,
                 ctarget=1000, layers=[200], batch_size=100, lr=0.001,
                 gamma=0.999, epsilon=0.01, epsilon_decay=0.99999,
                 lr_decay=1., device=device):

        self.replay_memory = ReplayMemory(replay_memory_capacity)
        self.replay_memory_capacity = replay_memory_capacity
        self.ctarget = ctarget
        self.n_actions = n_actions
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size

        # Learned network
        self.Q = Net(state_dim, n_actions,layers).to(device)
        # Target network
        self.Q_target = copy.deepcopy(self.Q).to(device)

        self.optimizer = optim.Adam(self.Q.parameters(), lr=lr)
        self.lr_sched = optim.lr_scheduler.ExponentialLR(self.optimizer, lr_decay)
        self.criterion = nn.SmoothL1Loss()

        self.lobs = None
        self.laction = None
        self.t = 0
        self.device = device
        self.lQvalue = None

    def act(self, obs, reward, done):

        obs = torch.tensor(obs).float().to(self.device)
        reward = torch.tensor(reward).float().to(self.device)

        if self.t > 0:
            self.replay_memory.push(self.lobs, self.laction, obs, reward)

        # epsilon greedy choice
        if random.random() < self.epsilon:
            action = random.randint(0, self.n_actions-1)
        else:
            with torch.no_grad():
                Q_value, action = torch.max(self.Q(obs.unsqueeze(0)),1)
            action = action.item()
            Q_value = Q_value.item()
            self.lQ_value = Q_value

        self.epsilon *= self.epsilon_decay
        self.lobs = obs
        self.laction = torch.tensor(action)
        return action

    def optimize(self, done):

        if self.t < self.batch_size:
            self.t += 1
            return 0.

        # sample random minibatch of transitions from memory
        transitions = self.replay_memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        # construct tensors
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
            dtype=torch.bool)
        non_final_next_states = torch.stack(
            [s for s in batch.next_state if s is not None]).float().to(device)

        state_batch = torch.stack(batch.state).float().to(device)
        action_batch = torch.stack(batch.action).to(device)
        reward_batch = torch.stack(batch.reward).float().to(device)

        # get output for batch
        # extract Q values only for played actions
        state_action_values = self.Q(state_batch).gather(1, action_batch.unsqueeze(1))

        next_state_values = torch.zeros(self.batch_size, device=device).float()
        with torch.no_grad():
            next_state_values[non_final_mask] = self.Q_target(non_final_next_states).max(1)[0]

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        # gradient descent step
        self.optimizer.step()
        # gradient clipping
        for param in self.Q.parameters():
            param.grad.data.clamp_(-1, 1)

        self.t += 1

        # every C step, reset target network
        if self.t % self.ctarget == 0:
            self.Q_target = copy.deepcopy(self.Q)

        if done:
            # Decay learning rate
            self.lr_sched.step()
        # if done:
        #     self.checkpoint.model = self.Q
        #     self.checkpoint.optimizer = self.optimizer

        return loss.item()
