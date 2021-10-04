import os
from shutil import rmtree
import numpy as np
import torch
from sklearn import metrics


def edge_auroc(pred_edges, true_edges):
    if true_edges.min() < 0 or true_edges.max() > 1:
        print("Groundtruth is CPDAG")
        true_edges = torch.clamp(true_edges, 0, 1)
    fpr, tpr, thresholds = metrics.roc_curve(true_edges.reshape(-1).cpu().detach().numpy(), pred_edges.reshape(-1).cpu().detach().numpy())
    return metrics.auc(fpr, tpr)


def edge_apr(pred_edges, true_edges):
    if true_edges.min() < 0 or true_edges.max() > 1:
        print("Groundtruth is CPDAG")
        true_edges = torch.clamp(true_edges, 0, 1)
    true_edges = torch.clamp(true_edges, 0, 1)
    return metrics.average_precision_score(true_edges.reshape(-1).cpu().detach().numpy(), pred_edges.reshape(-1).cpu().detach().numpy())


def edge_fn_fp_rev(pred_edges, true_edges):
    diff = true_edges - pred_edges
    rev = (((diff + diff.T) == 0) & (diff != 0)).sum() / 2.

    # Each reversed edge necessarily leads to one fp and one fn so we need to subtract those
    fn = (diff == 1).sum() - rev
    fp = (diff == -1).sum() - rev

    return fn.detach().cpu().numpy(), fp.detach().cpu().numpy(), rev.detach().cpu().numpy()
