import torch
from torch import FloatTensor
import torch.functional as F


class QNetwork(torch.nn.Module):
    """Model used for the dqn algorithm to estimate the Q values of environment states
    """
    num_inputs = 37
    num_actions = 4

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.layer1 = torch.nn.Linear(self.num_inputs, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.adv_layer = torch.nn.Linear(hidden_dim, self.num_actions)
        self.val_layer = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x: FloatTensor):
        """Forward pass through the model, input is x"""
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        adv = self.adv_layer(x)
        val = self.val_layer(x)
        q_val = adv + val  # adv = q_val - val
        return q_val
