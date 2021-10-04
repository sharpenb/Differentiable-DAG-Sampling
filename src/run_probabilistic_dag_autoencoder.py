import logging
import time
import torch
import os
import shutil
from sacred import Experiment
import seml
from seml.utils import flatten

from src.datasets.DAGDataset import get_dag_dataset
from src.probabilistic_dag_model.probabilistic_dag_autoencoder import ProbabilisticDAGAutoencoder
from src.probabilistic_dag_model.train_probabilistic_dag_autoencoder import train_autoencoder
from src.probabilistic_dag_model.test_probabilistic_dag_autoencoder import test_autoencoder

ex = Experiment()
seml.setup_logger(ex)

@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


@ex.automain
def run(
        # Dataset parameters
        seed_dataset,  # Seed to shuffle dataset. int
        dataset_name,  # Dataset name. string
        dataset_directory,  # Dataset name. string
        i_dataset,  # Dataset name. string
        split,  # Split for train/val/test sets. list of floats

        # Architecture parameters
        seed_model,  # Seed to init model. int
        directory_model,  # Path to save model. string
        ma_hidden_dims,
        ma_architecture,
        ma_fast,
        pd_initial_adj,
        pd_temperature,  # Temperature for differentiable sorting. List of ints
        pd_hard,  # Hard or soft sorting. List of ints
        pd_order_type,  # Type of differentiable sorting. List of ints
        pd_noise_factor,  # Noise factor for Sinkhorn sorting. List of ints

        # Training parameters
        directory_results,  # Path to save results. string
        max_epochs,  # Maximum number of epochs for training
        patience,  # Patience for early stopping. int
        frequency,  # Frequency for early stopping test. int
        batch_size,  # Batch size. int
        ma_lr,  # Learning rate. float
        pd_lr,  # Learning rate. float
        loss,  # Loss name. string
        regr,  # Regularization factor in Bayesian loss. float
        prior_p):  # Prior edge probability. float

    logging.info('Received the following configuration:')
    logging.info(f'DATASET | '
                 f'seed_dataset {seed_dataset} - '
                 f'dataset_name {dataset_name} - '
                 f'i_dataset {i_dataset} - '
                 f'split {split}')
    logging.info(f'ARCHITECTURE | '
                 f' seed_model {seed_model} - '
                 f' ma_hidden_dims {ma_hidden_dims} - '
                 f' ma_architecture {ma_architecture} - '
                 f' ma_fast {ma_fast} - '
                 f' pd_initial_adj {pd_initial_adj} - '
                 f' pd_temperature {pd_temperature} - '
                 f' pd_hard {pd_hard} - '
                 f' pd_order_type {pd_order_type} - '
                 f' pd_noise_factor {pd_noise_factor}')
    logging.info(f'TRAINING | '
                 f' max_epochs {max_epochs} - '
                 f' patience {patience} - '
                 f' frequency {frequency} - '
                 f' batch_size {batch_size} - '
                 f' ma_lr {ma_lr} - '
                 f' pd_lr {pd_lr} - '
                 f' loss {loss} - '
                 f' regr {regr} - '
                 f' prior_p {prior_p}')

    ##################
    ## Load dataset ##
    ##################
    train_loader, val_loader, test_loader, true_dag_adj = get_dag_dataset(dataset_directory=dataset_directory,
                                                                          dataset_name=dataset_name,
                                                                          i_dataset=i_dataset,
                                                                          batch_size=batch_size,
                                                                          split=split,
                                                                          seed=seed_dataset)
    input_dim = true_dag_adj.shape[0]
    if pd_initial_adj == 'GT':
        pd_initial_adjacency = true_dag_adj
    elif pd_initial_adj == 'RGT':
        pd_initial_adjacency = true_dag_adj.T
    elif pd_initial_adj == 'Random':
        pd_initial_adjacency = torch.rand_like(true_dag_adj)  # TODO make valid random DAG
    elif pd_initial_adj == 'Learned':
        pd_initial_adjacency = None

    #################
    ## Train model ##
    #################
    model = ProbabilisticDAGAutoencoder(input_dim=input_dim,
                                        output_dim=1,
                                        loss=loss,
                                        regr=regr,
                                        prior_p=prior_p,
                                        seed=seed_model,
                                        ma_hidden_dims=ma_hidden_dims,
                                        ma_architecture=ma_architecture,
                                        ma_lr=ma_lr,
                                        ma_fast=ma_fast,
                                        pd_temperature=pd_temperature,
                                        pd_hard=pd_hard,
                                        pd_order_type=pd_order_type,
                                        pd_noise_factor=pd_noise_factor,
                                        pd_initial_adj=pd_initial_adjacency,
                                        pd_lr=pd_lr)
    full_config_dict = flatten({'seed_dataset': seed_dataset,
                                'dataset_name': dataset_name,
                                'i_dataset': i_dataset,
                                'split': list(split),
                                'seed_model': seed_model,
                                'ma_hidden_dims': list(ma_hidden_dims),
                                'ma_architecture': ma_architecture,
                                'ma_fast': ma_fast,
                                'pd_initial_adj': pd_initial_adj,
                                'pd_temperature': pd_temperature,
                                'pd_hard': pd_hard,
                                'pd_order_type': pd_order_type,
                                'pd_noise_factor': pd_noise_factor,
                                'max_epochs': max_epochs,
                                'patience': patience,
                                'frequency': frequency,
                                'batch_size': batch_size,
                                'ma_lr': ma_lr,
                                'pd_lr': pd_lr,
                                'loss': loss,
                                'regr': regr,
                                'prior_p': prior_p})
    full_config_name = ''
    for k, v in full_config_dict.items():
        if isinstance(v, dict):
            v = flatten(v)
            v = [str(val) for key, val in v.items()]
            v = "-".join(v)
        full_config_name += str(v) + '-'
    full_config_name = full_config_name[:-1]
    model_path = f'{directory_model}/model-autopdag-{full_config_name}'

    t_start = time.time()
    train_losses, val_losses, train_mse, val_mse = train_autoencoder(model,
                                                                     train_loader,
                                                                     val_loader,
                                                                     max_epochs=max_epochs,
                                                                     frequency=frequency,
                                                                     patience=patience,
                                                                     model_path=model_path,
                                                                     full_config_dict=full_config_dict)
    t_end = time.time()

    ################
    ## Test model ##
    ################
    result_path = f'{directory_results}/results-autopdag-{full_config_name}'
    try:
        if not os.path.exists(result_path):
            os.makedirs(result_path)
    except OSError:
        print('Error: Creating directory. ' + result_path)
    model.load_state_dict(torch.load(f'{model_path}')['model_state_dict'])
    metrics = test_autoencoder(model, true_dag_adj, train_loader, test_loader, result_path, seed_dataset)
    shutil.rmtree(result_path)

    results = {
        'model_path': model_path,
        'result_path': result_path,
        'training_time': t_end - t_start,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_mse': train_mse,
        'val_mse': val_mse,
        'fail_trace': seml.evaluation.get_results
    }

    return {**results, **metrics}
