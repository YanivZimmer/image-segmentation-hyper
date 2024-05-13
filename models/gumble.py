import math
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class FeatureSelectorGumble(nn.Module):
    def __init__(self, input_dim, device, target_number=1, temp=0.01):
        super(FeatureSelectorGumble, self).__init__()  #
        self.target_number = target_number
        self.device = device
        self.input_dim = input_dim
        self.temp = temp
        params = torch.zeros(
            input_dim,
            device=self.device
        )
        params[random.randint(0, self.input_dim - 1)] = 0.5
        self.mu = torch.nn.Parameter(
            0.01
            * params + 0.5,
            requires_grad=True,
        )
        self.regularizer = lambda : 0

    def get_gates(self, mode):
        if self.mu is None:
            return None
        if mode == "raw":
            return self.mu.detach().cpu().numpy()
        elif mode == "prob":
            return np.minimum(
                1.0, np.maximum(0.0, self.mu.detach().cpu().numpy() + 0.5)
            )
        else:
            raise NotImplementedError()

    def forward(self, x):
        feature_probs = F.gumbel_softmax(self.mu, self.temp, hard=True)
        sampled_feature_idx = torch.multinomial(feature_probs, num_samples=1)
        # sampled_feature_value = self.feature_values[sampled_feature_idx]
        # print(x.shape,  x[:,:,sampled_feature_idx].shape)
        return x[:, sampled_feature_idx]
