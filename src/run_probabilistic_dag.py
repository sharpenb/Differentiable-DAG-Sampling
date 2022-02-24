import logging
import time
import torch
from sacred import Experiment
import seml

from src.datasets.DAGDataset import get_dag_dataset
from src.probabilistic_dag_model.probabilistic_dag import ProbabilisticDAG
from src.probabilistic_dag_model.train_dag import train
from src.probabilistic_dag_model.test_dag import test

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
        i_dataset,  # Dataset name. string
        split,  # Split for train/val/test sets. list of floats

        # Architecture parameters
        seed_model,  # Seed to init model. int
        directory_model,  # Path to save model. string
        temperature,  # Input dimension. List of ints
        hard,  # Input dimension. List of ints
        order_type,  # Input dimension. List of ints
        noise_factor,  # Input dimension. List of ints

        # Training parameters
        directory_results,  # Path to save resutls. string
        max_epochs,  # Maximum number of epochs for training. int
        patience,  # Patience for early stopping. int
        frequency,  # Frequency for early stopping test. int
        batch_size,  # Batch size. int
        lr,  # Learning rate. float
        loss,  # Loss name. string
        regr):  # Regularization factor in Bayesian loss. float

    logging.info('Received the following configuration:')
    logging.info(f'DATASET | '
                 f'seed_dataset {seed_dataset} - '
                 f'dataset_name {dataset_name} - '                 
                 f'i_dataset {i_dataset} - '
                 f'split {split}')
    logging.info(f'ARCHITECTURE | '
                 f' seed_model {seed_model} - '
                 f' temperature {temperature} - '
                 f' hard {hard} - '
                 f' order_type {order_type} - '
                 f' noise_factor {noise_factor}')
    logging.info(f'TRAINING | '
                 f' max_epochs {max_epochs} - '
                 f' patience {patience} - '
                 f' frequency {frequency} - '
                 f' batch_size {batch_size} - '
                 f' lr {lr} - '
                 f' loss {loss} - '
                 f' regr {regr}')

    ##################
    ## Load dataset ##
    ##################
    train_loader, val_loader, test_loader, true_dag_adj = get_dag_dataset(dataset_name=dataset_name,
                                                                          i_dataset=i_dataset,
                                                                          batch_size=batch_size,
                                                                          split=split,
                                                                          seed=seed_dataset)
    input_dim = true_dag_adj.shape[0]

    #################
    ## Train model ##
    #################
    if order_type=='sinkhorn' or order_type=='topk':
        model = ProbabilisticDAG(n_nodes=input_dim,
                                 temperature=temperature,
                                 hard=hard,
                                 order_type=order_type,
                                 noise_factor=noise_factor,
                                 initial_adj=None,
                                 lr=lr,
                                 seed=seed_model)
    else:
        raise NotImplementedError
    full_config_dict = {'seed_dataset': seed_dataset,
                        'dataset_name': dataset_name,
                        'i_dataset': i_dataset,
                        'split': list(split),
                        'seed_model': seed_model,
                        'temperature': temperature,
                        'hard': hard,
                        'order_type': order_type,
                        'noise_factor': noise_factor,
                        'max_epochs': max_epochs,
                        'patience': patience,
                        'frequency': frequency,
                        'batch_size': batch_size,
                        'lr': lr,
                        'loss': loss,
                        'regr': regr}
    full_config_name = ''
    for k, v in full_config_dict.items():
        full_config_name += str(v) + '-'
    full_config_name = full_config_name[:-1]
    model_path = f'{directory_model}/model-pdag-{full_config_name}'

    t_start = time.time()
    model, prob_abs_losses, sampled_mse_losses = train(model,
                                                       true_dag_adj,
                                                       max_epochs=max_epochs,
                                                       frequency=frequency,
                                                       patience=patience,
                                                       model_path=model_path,
                                                       full_config_dict=full_config_dict)
    t_end = time.time()

    ################
    ## Test model ##
    ################
    result_path = f'{directory_results}/results-pdag-{full_config_name}'
    # model.load_state_dict(torch.load(f'{model_path}')['model_state_dict'])
    metrics = test(model, true_dag_adj)

    results = {
        'model_path': model_path,
        'result_path': result_path,
        'training_time': t_end - t_start,
        'prob_abs_losses': prob_abs_losses,
        'sampled_mse_losses': sampled_mse_losses,
        'fail_trace': seml.evaluation.get_results
    }

    return {**results, **metrics}
