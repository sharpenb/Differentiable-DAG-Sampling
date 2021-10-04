import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax
from src.probabilistic_dag_model.soft_sort import SoftSort_p1, gumbel_sinkhorn

# ------------------------------------------------------------------------------

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class ProbabilisticDAG(nn.Module):

    def __init__(self, n_nodes, temperature=1.0, hard=True, order_type='sinkhorn', noise_factor=1.0, initial_adj=None, lr=1e-3, seed=0):
        """Base Class for Probabilistic DAG Generator based on topological order sampling

        Args:
            n_nodes (int): Number of nodes
            temperature (float, optional): Temperature parameter for order sampling. Defaults to 1.0.
            seed (int, optional): Random seed. Defaults to 0.
            hard (bool, optional): If True output hard DAG. Defaults to True.
            initial_adj (torch.tensor, optional): Initial binary adjecency matrix from e.g. PNS.
                Edges with value 0 will not be learnt in further process Defaults to None.
        """
        super().__init__()

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.n_nodes = n_nodes
        self.temperature = temperature
        self.hard = hard
        self.order_type = order_type

        # Mask for ordering
        self.mask = torch.triu(torch.ones(self.n_nodes, self.n_nodes, device=device), 1)

        # define initial parameters
        if self.order_type == 'sinkhorn':
            self.noise_factor = noise_factor
            p = torch.zeros(n_nodes, n_nodes, requires_grad=True, device=device)
            self.perm_weights = torch.nn.Parameter(p)
        elif self.order_type == 'topk':
            p = torch.zeros(n_nodes, requires_grad=True, device=device)
            self.perm_weights = torch.nn.Parameter(p)
            self.sort = SoftSort_p1(hard=self.hard, tau=self.temperature)
        else:
            raise NotImplementedError
        e = torch.zeros(n_nodes, n_nodes, requires_grad=True, device=device)
        torch.nn.init.uniform_(e)
        if initial_adj is not None:
            initial_adj = initial_adj.to(device)
            zero_indices = (1 - initial_adj).bool()
            # set masked edges to zero probability
            e.requires_grad = False
            e[zero_indices] = -300
            e.requires_grad = True
        torch.diagonal(e).fill_(-300)
        self.edge_log_params = torch.nn.Parameter(e)
        if initial_adj is not None:
            self.edge_log_params.register_hook(lambda grad: grad * initial_adj.float())

        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def sample_edges(self):
        p_log = F.logsigmoid(torch.stack((self.edge_log_params, -self.edge_log_params)))
        dag = gumbel_softmax(p_log, hard=True, dim=0)[0]
        return dag

    def sample_permutation(self):
        if self.order_type == 'sinkhorn':
            log_alpha = F.logsigmoid(self.perm_weights)
            P, _ = gumbel_sinkhorn(log_alpha, noise_factor=self.noise_factor, temp=self.temperature, hard=self.hard)
            P = P.squeeze().to(device)
        elif self.order_type == 'topk':
            logits = F.log_softmax(self.perm_weights, dim=0).view(1, -1)
            gumbels = -torch.empty_like(logits).exponential_().log()
            gumbels = (logits + gumbels) / 1
            P = self.sort(gumbels)
            P = P.squeeze()
        else:
            raise NotImplementedError
        return P

    def sample(self):
        P = self.sample_permutation()
        P_inv = P.transpose(0, 1)
        dag_adj = self.sample_edges()
        dag_adj = dag_adj * torch.matmul(torch.matmul(P_inv, self.mask), P)  # apply autoregressive masking
        return dag_adj

    def log_prob(self, dag_adj):
        raise NotImplementedError

    def deterministic_permutation(self, hard=True):
        if self.order_type == 'sinkhorn':
            log_alpha = F.logsigmoid(self.perm_weights)
            P, _ = gumbel_sinkhorn(log_alpha, temp=self.temperature, hard=hard, noise_factor=0)
            P = P.squeeze().to(device)
        elif self.order_type == 'topk':
            sort = SoftSort_p1(hard=hard, tau=self.temperature)
            P = sort(self.perm_weights.detach().view(1, -1))
            P = P.squeeze()
        return P

    def get_threshold_mask(self, threshold):
        P = self.deterministic_permutation()
        P_inv = P.transpose(0, 1)
        dag = (torch.sigmoid(self.edge_log_params.detach()) > threshold).float()
        dag = dag * torch.matmul(torch.matmul(P_inv, self.mask), P)  # apply autoregressive masking
        return dag

    def get_prob_mask(self):
        P = self.deterministic_permutation()
        P_inv = P.transpose(0, 1)
        e = torch.sigmoid(self.edge_log_params.detach())
        e = e * torch.matmul(torch.matmul(P_inv, self.mask), P)  # apply autoregressive masking
        return e

    def print_parameters(self, prob=True):
        print('Permutation Weights')
        print(torch.sigmoid(self.perm_weights) if prob else self.perm_weights)
        print('Edge Probs')
        print(torch.sigmoid(self.edge_log_params) if prob else self.edge_log_params)
