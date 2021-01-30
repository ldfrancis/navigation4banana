import torch
import torch.functional as F

class QNetwork(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Forward pass through the model, input is x"""
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        adv = self.adv_layer(x)
        val = self.val_layer(x)
        q_val = adv + val # adv = q_val - val
        return q_val
