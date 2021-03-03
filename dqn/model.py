import torch
from torch import FloatTensor
import torch.nn.functional as F

from config import NUM_OBS, NUM_ACT, HIDDEN_DIM


class QNetwork(torch.nn.Module):
    """Model used for the dqn algorithm to estimate the Q values of environment states
    """

    def __init__(self):
        super().__init__()
        self.hidden1 = torch.nn.Linear(NUM_OBS, HIDDEN_DIM[0])
        self.hidden = torch.nn.Sequential(*[torch.nn.Linear(HIDDEN_DIM[i], HIDDEN_DIM[i+1])
                                            for i in range(len(HIDDEN_DIM)-1)])
        self.q_layer = torch.nn.Linear(HIDDEN_DIM[-1], NUM_ACT)

    def forward(self, x: FloatTensor):
        """Forward pass through the model, input is x"""
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden(x)
        x = F.relu(x)
        q_val =self.q_layer(x)  # adv = q_val - val
        return q_val
