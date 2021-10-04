import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
import math


class ParallelLinear(nn.Module):
    def __init__(self, input_dim, output_dim, parallel_dim):
        """
        Class which applies `parallel_dim` different linear transformations in a vectorized way.
        :param input_dim: input shape of a single linear transformation.
        :param output_dim: output shape of a single linear tranformation.
        :param parallel_dim: number of parallel linear transformations
        """
        super().__init__()
        self.input_dim = input_dim
        self.parallel_dim = parallel_dim
        self.output_dim = output_dim
        self.weight = Parameter(torch.Tensor(parallel_dim, output_dim, input_dim))   # [parallel_dim, output_dim, input_dim]
        self.bias = Parameter(torch.Tensor(parallel_dim, 1, output_dim))   # [parallel_dim, 1, output_dim]
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        """
        :param input: inut to transform. shape: [parallel_dim, batch_size, input_dim]
        :return: output after linear transformations. Each `parallel_dim` dimension represent a different linear transformation.
        These transformations are shared across the batch_size. shape: [parallel_dim, batch_size, output_dim]
        """
        output = torch.bmm(input, self.weight.transpose(2, 1))
        output += self.bias
        # output = torch.einsum("tij,btj->bti", self.weight, input) + self.bias
        return output
