import torch
import numpy as np
from src.metrics.dag_metrics import edge_auroc, edge_apr, edge_fn_fp_rev

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


def test_autoencoder(model, true_dag_adj, train_loader, test_loader, result_path, seed_dataset):
    model.eval()

    # Structure metrics
    if model.pd_initial_adj is None: # DAG is learned
        prob_mask = model.probabilistic_dag.get_prob_mask()
    else: # DAG is fixed
        prob_mask = model.pd_initial_adj

    # DAG learning
    metrics = {'undirected_edge_auroc': edge_auroc(pred_edges=prob_mask + prob_mask.T, true_edges=true_dag_adj + true_dag_adj.T),
               'undirected_edge_apr': edge_apr(pred_edges=prob_mask + prob_mask.T, true_edges=true_dag_adj + true_dag_adj.T),
               'reverse_edge_auroc': edge_auroc(pred_edges=prob_mask, true_edges=true_dag_adj.T),
               'reverse_edge_apr': edge_apr(pred_edges=prob_mask, true_edges=true_dag_adj.T),
               'edge_auroc': edge_auroc(pred_edges=prob_mask, true_edges=true_dag_adj),
               'edge_apr': edge_apr(pred_edges=prob_mask, true_edges=true_dag_adj),
               }

    # Causal mechanims learning
    with torch.no_grad():
        for batch_index, (X) in enumerate(test_loader):
            X = X.to(device)
            model.update_mask(type='deterministic')
            X_pred = model(X)
            if batch_index == 0:
                X_pred_all = X_pred.reshape(-1).to("cpu")
                X_all = X.reshape(-1).to("cpu")
            else:
                X_pred_all = torch.cat([X_pred_all, X_pred.reshape(-1).to("cpu")], dim=0)
                X_all = torch.cat([X_all, X.reshape(-1).to("cpu")], dim=0)
        reconstruction = ((X_all - X_pred_all)**2).mean().item()

    metrics['reconstruction'] = reconstruction
    return metrics
