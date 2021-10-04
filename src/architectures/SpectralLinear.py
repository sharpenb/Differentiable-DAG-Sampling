import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm, remove_spectral_norm


class SpectralLinear(nn.Module):
    def __init__(self, input_dim, output_dim, k_lipschitz=1.0):
        super().__init__()
        self.k_lipschitz = k_lipschitz
        self.spectral_linear = spectral_norm(nn.Linear(input_dim, output_dim))
        # self.spectral_linear = spectral_norm(nn.Linear(input_dim, output_dim))
        # self.spectral_linear(torch.Tensor(torch.ones(1, input_dim)))
        # remove_spectral_norm(self.spectral_linear)
        # self.spectral_linear.weight = nn.Parameter(self.spectral_linear.weight * self.k_lipschitz)

    def forward(self, x):
        y = self.k_lipschitz * self.spectral_linear(x)
        # y = self.spectral_linear(x)
        return y
