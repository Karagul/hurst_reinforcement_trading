import numpy as np
import torch
import torch.nn as nn
from torch import optim
import abc

class BaseModel(nn.Module):
    def __init__(self,
                 dim_observation,
                 n_actions,
                 hidden_dims=(8,8),
                 n_layers=2,
                 use_cuda=False,
                ):
        super(BaseModel, self).__init__()

        self.n_actions = n_actions
        self.dim_observation = dim_observation
        self.use_cuda = use_cuda

        # to implement in child classes : self.net

    @property
    def dtype(self):
        if self.use_cuda:
            return torch.cuda.FloatTensor
        else:
            return torch.FloatTensor

    def forward(self, state):
        return self.net(state)

    def select_action(self, state):
        action = torch.multinomial(self.forward(state), 1)
        return action

class MLPModel(BaseModel):
    def __init__(self,
                 dim_observation,
                 n_actions,
                 use_cuda=False,
                ):
        super().__init__(dim_observation,
                         n_actions,
                         use_cuda=use_cuda,
                        )

        self.net = nn.Sequential(
            nn.Linear(in_features=self.dim_observation, out_features=64),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=8),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=self.n_actions),
            nn.Softmax(dim=-1)
        )

class MLPAutocorrModel(BaseModel):
    def __init__(self,
                 dim_observation,
                 n_actions,
                 use_cuda=False,
                ):
        super().__init__(dim_observation,
                         n_actions,
                         use_cuda=use_cuda,
                        )

        self.net = nn.Sequential(
            nn.Linear(in_features=self.dim_observation, out_features=12),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(in_features=12, out_features=self.n_actions),
            nn.Softmax(dim=-1)
        )
