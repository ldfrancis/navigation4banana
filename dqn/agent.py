from collections import defaultdict
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F

from .model import QNetwork
from .buffer import ReplayBuffer
from .constants import LR, GAMMA, TAU


class DQN:

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, num_actions, state_size):
        self.num_actions = num_actions
        self.q_network = QNetwork(num_actions, state_size)
        self.q_network_target = QNetwork(num_actions, state_size)
        self.buffer = ReplayBuffer()
        self.policy = defaultdict(lambda : (1/num_actions)*np.ones((num_actions,)))
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=LR)

    def take_action(self, state:np.ndarray, epsilon=0):
        # obtain q values
        state = torch.FloatTensor(state)
        q_vals = self.q_network(state)

        # obtain greedy action
        max_action = np.argmax(q_vals.cpu().numpy())

        # set probs for policy
        non_greedy_prob = epsilon/self.num_actions
        greedy_prob = 1-(self.num_actions-1)*non_greedy_prob
        policy = [non_greedy_prob]*self.num_actions
        policy[max_action] = greedy_prob

        # sample action
        action = np.random.choice(self.num_actions, 1, p=policy)

        # update q networks
        self.update_q_networks()
        return action

    def save_transition(self, transition:Tuple):
        self.buffer.add(*transition)

    def update_q_networks(self):
        state, action, rew, next_state, done = self.buffer.sample()
        td_target = rew + GAMMA*(1-done)*self.q_network_target(next_state).detach().max(-1)
        q_val = self.q_network(state).gather(-1, action)
        td_error = F.mse_loss(q_val, td_target)
        self.optimizer.zero_grad()
        td_error.backward()
        self.optimizer.step()
        # update target network
        for target_param, local_param in zip(self.q_network_target.parameters(), self.q_network.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)



