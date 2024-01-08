import math

import numpy as np
import torch
from torch import nn


class EHBSFeatureSelector(nn.Module):
    def __init__(self, input_dim, sigma, device, headstart_idx=None):
        super(EHBSFeatureSelector, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.mu = torch.nn.Parameter(
            0.01
            * torch.randn(
                input_dim,
            ),
            requires_grad=True,
        )
        self.noise = torch.randn(self.mu.size(),device=device)
        self.sigma = sigma

    def apply_ndim_mask(self, mask_1d: torch.Tensor, x: torch.Tensor):
        mask = mask_1d.view(1, 1, -1, 1, 1).expand(*x.shape)
        # mask = np.tile(mask_1d[np.newaxis,np.newaxis,:, np.newaxis, np.newaxis], x.shape)
        return x.to(self.device) * mask.to(self.device)

    def apply_mask_loop(self, mask_1d: torch.Tensor, x: torch.Tensor):
        # batch,bands,p,p <- x.shape
        # batch p,p,bands -> x.shape
        x = x.squeeze()
        x = torch.transpose(x, 1, 3)
        x = x * mask_1d
        x = torch.transpose(x, 1, 3)
        batch_size, bands, p, p = x.shape
        # for idx in range(x.shape[0]):
        #    x[idx] = (x[idx].T*mask_1d).T
        return x

    def forward(self, x):
        discount = 1
        z = self.mu + discount*self.sigma * self.noise.normal_() * self.training
        stochastic_gate = self.hard_sigmoid(z)
        if len(x.shape) == 2:
            return x * stochastic_gate
        temp = torch.ones(stochastic_gate.shape)
        temp[1] = 0
        y=torch.Tensor(x)
        x = x.squeeze()
        x = torch.transpose(x, 0, -1)
        x = x * stochastic_gate
        x = torch.transpose(x, 0, -1)
        return x.unsqueeze(0)

    def hard_sigmoid(self, x):
        return torch.clamp(x + 0.5, 0.0, 1.0)

    def regularizer_formula(self, x):
        # if self.const_masking is not None:
        #    return torch.Tensor([0])
        """Gaussian CDF."""
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    def regularizer(self):
        return torch.mean(self.regularizer_formula((self.mu + 0.5) / self.sigma))

    # def _apply(self, fn):
    #     super(FeatureSelector, self)._apply(fn)
    #     self.noise = fn(self.noise)
    #     return self
    #
    # def set_mask(self, mask):
    #     self.mask = mask

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

