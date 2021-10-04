import numpy as np
import torch.nn as nn
from src.architectures.ParallelLinear import ParallelLinear


def parallel_linear_sequential(input_dims, hidden_dims, output_dim, parallel_dim, p_drop=None):
    """
    Create a sequential of ParallelLinear.
    :param input_dims: input dimension. List of int.
    :param hidden_dims: hidden dimensions. List of int.
    :param output_dim: output_dimension. List of int.
    :param parallel_dim: number of parallel transformations. int
    :param p_drop: drop out probability. If None, no drop out. float (default: None)
    :return: sequential layers of parallel transformations.
    """
    dims = [np.prod(input_dims)] + hidden_dims + [output_dim]
    num_layers = len(dims) - 1
    layers = []
    for i in range(num_layers):
        layers.append(ParallelLinear(dims[i], dims[i + 1], parallel_dim))
        if i < num_layers - 1:
            layers.append(nn.ReLU())
            if p_drop is not None:
                layers.append(nn.Dropout(p=p_drop))
    return nn.Sequential(*layers)
