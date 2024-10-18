import torch
import torch.nn as nn
import torch.nn.functional

import torch

def DTW_calc(ts1, ts2):
    """
    Computes the DTW similarity (for univariate time series) between two time series.
    Handles one input having batch size 1 for broadcasting (shapelet of shape (1, len_shapelet)).
    
    Parameters:
    ts1: Tensor of shape (batch_size1, time_steps1) or (1, time_steps1) (shapelet)
    ts2: Tensor of shape (batch_size2, time_steps2) (time series segment)
    
    Returns:
    dtw_dist: Tensor of shape (batch_size,) containing the DTW distance for each batch.
    """
    # Compute the cost matrix
    cost_mat = cost_matrix(ts1, ts2)

    # Return the square root of the final element in the cost matrix (total DTW cost for each batch)
    return torch.sqrt(cost_mat[:, -1, -1])

def euclidean_diff(ts1, ts2):
    """
    Calculates the squared Euclidean distance between two points (or vectors) from two univariate time series.
    Handles broadcasting if one input has batch size 1 (shapelet of shape (1,)).
    
    Parameters:
    ts1: Tensor of shape (batch_size,)
    ts2: Tensor of shape (batch_size,) or (1,)
    
    Returns:
    dist: Tensor of shape (batch_size,) containing the squared Euclidean distance for each batch.
    """
    return (ts1 - ts2) ** 2

def cost_matrix(ts1, ts2):
    """
    Computes the cost matrix for DTW with differentiable operations.
    Handles one input having batch size 1 (for shapelet broadcasting), supports univariate time series.
    
    Parameters:
    ts1: Tensor of shape (batch_size, time_steps1) or (1, time_steps1)
    ts2: Tensor of shape (batch_size, time_steps2)
    
    Returns:
    cum_sum: Tensor of shape (batch_size, time_steps1, time_steps2) containing the cumulative sum for each batch.
    """
    batch_size1, ts1_size = ts1.shape 
    batch_size2, ts2_size = ts2.shape


    batch_size = max(batch_size1, batch_size2)


    cum_sum = torch.zeros((batch_size, ts1_size + 1, ts2_size + 1), device=ts1.device)
    cum_sum[:, 1:, 0] = float('inf')
    cum_sum[:, 0, 1:] = float('inf')


    for i in range(1, ts1_size + 1):
        for j in range(1, ts2_size + 1):
            cost = euclidean_diff(ts1[:, i - 1], ts2[:, j - 1])
            cum_sum[:, i, j] = cost + torch.min(torch.stack([
                cum_sum[:, i - 1, j],   # Insertion
                cum_sum[:, i, j - 1],   # Deletion
                cum_sum[:, i - 1, j - 1]  # Match
            ], dim=0), dim=0)[0]

    return cum_sum[:, 1:, 1:]

