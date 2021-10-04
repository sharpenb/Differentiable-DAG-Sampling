import numpy as np
import networkx as nx
import torch
import torch.utils.data as td
from torch.utils.data.sampler import SubsetRandomSampler
from cdt.data import load_dataset

synthetic_datasets = set(['data_p100_e100_n1000_GP',
                          'data_p100_e400_n1000_GP',
                          'data_p10_e10_n1000_GP',
                          'data_p10_e40_n1000_GP',
                          'data_p20_e20_n1000_GP',
                          'data_p20_e80_n1000_GP',
                          'data_p50_e200_n1000_GP',
                          'data_p50_e50_n1000_GP',
                          'data_sf_p100_e100_n1000_GP',
                          'data_sf_p100_e400_n1000_GP',
                          'data_sf_p10_e10_n1000_GP',
                          'data_sf_p10_e40_n1000_GP',
                          'data_sf_p20_e20_n1000_GP',
                          'data_sf_p20_e80_n1000_GP',
                          'data_sf_p50_e200_n1000_GP',
                          'data_sf_p50_e50_n1000_GP',
                          ])


def get_dag_dataset(dataset_directory, dataset_name, i_dataset, batch_size, split=[.6, .2, .2], seed=1):

    assert np.sum(split) == 1.0
    np.random.seed(seed)

    if dataset_name in synthetic_datasets:
        train_loader, val_loader, test_loader, dag_adj = DAGDataset.synthetic(dataset_directory,dataset_name, i_dataset, batch_size=batch_size, split=split)
    elif dataset_name == 'sachs':
        train_loader, val_loader, test_loader, dag_adj = DAGDataset.sachs(batch_size=batch_size, split=split)
    elif dataset_name == 'syntren':
        train_loader, val_loader, test_loader, dag_adj = DAGDataset.syntren(dataset_directory, i_dataset, batch_size=batch_size, split=split)
    else:
        raise NotImplementedError

    return train_loader, val_loader, test_loader, dag_adj


class DAGDataset:
    """
    The dataset class provides static methods to use different datasets and their splits for
    different seeds.
    """

    @classmethod
    def synthetic(cls, dataset_directory, dataset_name, i_dataset, batch_size, split):
        """
        Synthetic dataset
        """
        # Load the graph and data
        adjacency = np.load(f'{dataset_directory}/{dataset_name}/DAG{i_dataset}.npy')
        dag_adj = torch.as_tensor(adjacency).type(torch.Tensor)
        X = np.load(f'{dataset_directory}/{dataset_name}/data{i_dataset}.npy')
        X = torch.as_tensor(X).type(torch.Tensor)

        n_data = X.shape[0]
        indices = list(range(n_data))
        split0, split1 = int(n_data * split[0]), int(n_data * (split[0] + split[1]))
        np.random.shuffle(indices)

        # Train split
        train_indices = indices[:split0]
        mean, std = torch.mean(X[train_indices], 0, keepdim=True), torch.std(X[train_indices], 0, keepdim=True)
        X_scaled = (X - mean) / std
        train_sampler = SubsetRandomSampler(train_indices)
        train_loader = torch.utils.data.DataLoader(X_scaled, batch_size=batch_size, sampler=train_sampler, num_workers=8)
        # Validation split
        val_indices = indices[split0:split1]
        val_sampler = SubsetRandomSampler(val_indices)
        val_loader = torch.utils.data.DataLoader(X_scaled, batch_size=1024, sampler=val_sampler, num_workers=8)
        # Test split
        test_indices = indices[split1:]
        test_sampler = SubsetRandomSampler(test_indices)
        test_loader = torch.utils.data.DataLoader(X_scaled, batch_size=1024, sampler=test_sampler, num_workers=8)

        return train_loader, val_loader, test_loader, dag_adj


    @classmethod
    def sachs(cls, batch_size, split):
        """
        Sachs dataset
        """
        # Load the graph and data
        data, graph = load_dataset("sachs")
        adjacency = nx.to_numpy_array(graph)
        dag_adj = torch.as_tensor(adjacency).type(torch.Tensor)
        X = data.values[:853]
        # X = data.values
        X = torch.as_tensor(X).type(torch.Tensor)

        n_data = X.shape[0]
        indices = list(range(n_data))
        split0, split1 = int(n_data * split[0]), int(n_data * (split[0] + split[1]))
        np.random.shuffle(indices)

        # Train split
        train_indices = indices[:split0]
        mean, std = torch.mean(X[train_indices], 0, keepdim=True), torch.std(X[train_indices], 0, keepdim=True)
        X_scaled = (X - mean) / std
        train_sampler = SubsetRandomSampler(train_indices)
        train_loader = torch.utils.data.DataLoader(X_scaled, batch_size=batch_size, sampler=train_sampler, num_workers=8)
        # Validation split
        val_indices = indices[split0:split1]
        val_sampler = SubsetRandomSampler(val_indices)
        val_loader = torch.utils.data.DataLoader(X_scaled, batch_size=1024, sampler=val_sampler, num_workers=8)
        # Test split
        test_indices = indices[split1:]
        test_sampler = SubsetRandomSampler(test_indices)
        test_loader = torch.utils.data.DataLoader(X_scaled, batch_size=1024, sampler=test_sampler, num_workers=8)

        return train_loader, val_loader, test_loader, dag_adj

    @classmethod
    def syntren(cls, dataset_directory, i_dataset, batch_size, split):
        """
        SynTReN dataset
        """
        # Load the graph and data
        adjacency = np.load(f'{dataset_directory}/syntren/DAG{i_dataset}.npy')
        dag_adj = torch.as_tensor(adjacency).type(torch.Tensor)
        X = np.load(f'{dataset_directory}/syntren/data{i_dataset}.npy')
        X = torch.as_tensor(X).type(torch.Tensor)

        n_data = X.shape[0]
        indices = list(range(n_data))
        split0, split1 = int(n_data * split[0]), int(n_data * (split[0] + split[1]))
        np.random.shuffle(indices)

        # Train split
        train_indices = indices[:split0]
        mean, std = torch.mean(X[train_indices], 0, keepdim=True), torch.std(X[train_indices], 0, keepdim=True)
        X_scaled = (X - mean) / std
        train_sampler = SubsetRandomSampler(train_indices)
        train_loader = torch.utils.data.DataLoader(X_scaled, batch_size=batch_size, sampler=train_sampler, num_workers=8)
        # Validation split
        val_indices = indices[split0:split1]
        val_sampler = SubsetRandomSampler(val_indices)
        val_loader = torch.utils.data.DataLoader(X_scaled, batch_size=1024, sampler=val_sampler, num_workers=8)
        # Test split
        test_indices = indices[split1:]
        test_sampler = SubsetRandomSampler(test_indices)
        test_loader = torch.utils.data.DataLoader(X_scaled, batch_size=1024, sampler=test_sampler, num_workers=8)

        return train_loader, val_loader, test_loader, dag_adj
