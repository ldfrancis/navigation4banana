from collections import defaultdict
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F

import os

from .model import QNetwork
from .buffer import ReplayBuffer
from config import LR, GAMMA, TAU, BATCH_SIZE, NUM_ACT
from pathlib import Path


class DQN:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        self.q_network = QNetwork()
        self.q_network_target = QNetwork()
        self.buffer = ReplayBuffer()
        self.policy = defaultdict(lambda : (1/NUM_ACT)*np.ones((NUM_ACT,)))
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=LR)

        # restore if checkpoint exists
        if Path("./checkpoint/checkpoint.pt").exists():
            self.restore("./checkpoint/checkpoint.pt")

    def take_action(self, state:np.ndarray, train=False, epsilon=0):
        # obtain q values
        state = torch.FloatTensor(state[None, :]).to(DQN.device)
        q_vals = self.q_network(state)

        # obtain greedy action
        max_action = np.argmax(q_vals.detach().cpu().numpy())

        # set probs for policy
        non_greedy_prob = epsilon/NUM_ACT
        greedy_prob = 1-(NUM_ACT-1)*non_greedy_prob
        policy = [non_greedy_prob]*NUM_ACT
        policy[max_action] = greedy_prob

        # sample action
        action = np.random.choice(NUM_ACT, 1, p=policy)

        # update q networks
        if train:
            self.update_q_networks()
        return action

    def save_transition(self, transition:Tuple):
        self.buffer.add(*transition)

    def update_q_networks(self):
        if len(self.buffer) < BATCH_SIZE:
            return
        state, action, rew, next_state, done = self.buffer.sample()
        state = torch.FloatTensor(state).to(DQN.device)
        action = torch.Tensor(action).long().to(DQN.device)
        rew = torch.FloatTensor(rew).to(DQN.device)
        next_state = torch.FloatTensor(next_state).to(DQN.device)
        done = torch.from_numpy(done).float().to(DQN.device)
        td_target = (rew + GAMMA*(1-done)*self.q_network_target(next_state).detach().max(-1)[0]).unsqueeze(-1)
        q_val = self.q_network(state).gather(-1, action)
        td_error = F.mse_loss(q_val, td_target)
        self.optimizer.zero_grad()
        td_error.backward()
        self.optimizer.step()

        # update target network
        for target_param, local_param in zip(self.q_network_target.parameters(), self.q_network.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

    def save(self, path="./checkpoint"):
        os.makedirs(path, exist_ok=True)
        self.checkpoint_path = path
        self.checkpoint_path_file = f"{path}/checkpoint.pt"
        torch.save({
            'q_state_dict': self.q_network.state_dict(),
            'q_target_state_dict': self.q_network_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, self.checkpoint_path_file )

    def restore(self, path):
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_state_dict'])
        self.q_network_target.load_state_dict(checkpoint['q_target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])





