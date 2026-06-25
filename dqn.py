import torch
import torch.nn as nn

import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim=12, action_dim=2, hidden_dim=512):
        super(DQN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(0.1), # Stabilizes coordinate processing better than plain ReLU
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Explicit weight initialization to prevent early blowups
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=0.1)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.model(x)