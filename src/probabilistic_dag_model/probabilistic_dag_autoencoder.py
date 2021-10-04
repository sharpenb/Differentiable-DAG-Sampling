import numpy as np
import torch
from torch import nn

from src.probabilistic_dag_model.masked_autoencoder import MaskedAutoencoder
from src.probabilistic_dag_model.masked_autoencoder import MaskedAutoencoderFast
from src.probabilistic_dag_model.probabilistic_dag import ProbabilisticDAG

masked_autoencoder = {True: MaskedAutoencoderFast,
                      False: MaskedAutoencoder}


class ProbabilisticDAGAutoencoder(nn.Module):
    def __init__(self,
                 # General parameters
                 input_dim,  # Input dimension (i.e. number of input signals). list of ints
                 output_dim,  # Output dimension (i.e. dimension of one input signal). list of ints
                 loss='ELBO',  # Loss name. string
                 regr=0,  # Regularization factor in ELBO loss. float
                 prior_p=.001,  # Regularization factor in ELBO loss. float
                 seed=123,

                 # Mask autoencoder parameters
                 ma_hidden_dims=[64, 64, 64],  # Hidden dimensions. list of ints
                 ma_architecture='linear',  # Encoder architecture name. int
                 ma_lr=1e-3,  # Learning rate. float
                 ma_fast=False,  # Use fast masked autoencoder implementation. Boolean

                 # Probabilistic dag parameters
                 pd_temperature=1.0,
                 pd_hard=True,
                 pd_order_type='sinkhorn',
                 pd_noise_factor=1.0,
                 pd_initial_adj=None,
                 pd_lr=1e-3):  # Random seed for init. int
        super().__init__()

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # torch.set_default_tensor_type(torch.FloatTensor)

        self.loss, self.regr, self.prior_p = loss, regr, prior_p

        # Autoencoder parameters
        self.mask_autoencoder = masked_autoencoder[ma_fast](input_dim=input_dim,
                                                            output_dim=output_dim,
                                                            hidden_dims=ma_hidden_dims,
                                                            architecture=ma_architecture,
                                                            lr=ma_lr,
                                                            seed=seed)

        # Probabilistic DAG parameters
        if pd_order_type == 'sinkhorn' or pd_order_type == 'topk':
            self.probabilistic_dag = ProbabilisticDAG(n_nodes=input_dim,
                                                      temperature=pd_temperature,
                                                      hard=pd_hard,
                                                      order_type=pd_order_type,
                                                      noise_factor=pd_noise_factor,
                                                      initial_adj=pd_initial_adj,
                                                      lr=pd_lr,
                                                      seed=seed)
        else:
            raise NotImplementedError
        self.pd_initial_adj = pd_initial_adj

    def forward(self, X, compute_loss=True):
        X_pred = self.mask_autoencoder(X)

        # Loss
        if compute_loss:
            if self.loss == 'ELBO':
                self.grad_loss = self.ELBO_loss(X_pred, X)
            else:
                raise NotImplementedError

        return X_pred

    def update_mask(self, type=None, threshold=.5):
        if type == 'deterministic':
            new_mask = self.probabilistic_dag.get_threshold_mask(threshold)
        elif type == 'id':
            new_mask = torch.eye(self.mask_autoencoder.input_dim)
        else:
            new_mask = self.probabilistic_dag.sample()
        if self.pd_initial_adj is not None:  # We do no sample if the DAG adjacency is not learned/specified from start
            new_mask = self.pd_initial_adj.to(next(self.mask_autoencoder.parameters()).device)
        self.mask_autoencoder.mask = new_mask

    def ELBO_loss(self, X_pred, X):
        loss = nn.MSELoss(reduction='mean')
        ELBO_loss = loss(X_pred, X)

        if self.regr > 0:
            kl_loss = torch.nn.KLDivLoss(reduction='mean')
            regularizer = kl_loss(self.probabilistic_dag.edge_log_params, self.prior_p * torch.ones_like(self.probabilistic_dag.edge_log_params))
            # p_log = F.logsigmoid(torch.stack((self.probabilistic_dag.edge_log_params, -self.probabilistic_dag.edge_log_params)))
            # regularizer = kl_loss(self.probabilistic_dag.edge_log_params, self.prior_p * torch.ones_like(self.probabilistic_dag.edge_log_params))
            ELBO_loss = ELBO_loss + self.regr * regularizer

        return ELBO_loss

    def step(self):
        self.mask_autoencoder.optimizer.zero_grad()
        if self.pd_initial_adj is None:
            self.probabilistic_dag.optimizer.zero_grad()
        self.grad_loss.backward()
        self.mask_autoencoder.optimizer.step()
        if self.pd_initial_adj is None:
            self.probabilistic_dag.optimizer.step()
