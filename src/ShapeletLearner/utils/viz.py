import torch
import matplotlib.pyplot as plt

from .DTW import *  # Although I haven't implemented this yet

def plot_shapelets(shapelets, timeseries):
    """
    Plot the top k best Learned Shapelets for a given timeseries that contribute to a specific outcome that we find contributes 
    the most. 
    
    Arguments:
        shapelets: List of learned shapelets.
        time_series: List of time series data.
    """

    fig, axes = plt.subplots(10, 1, figsize = (10, 32))
    
    for i,shapelet in enumerate(shapelets):
        shapelet_length = len(shapelet)
        
        distances = []
        for j in range(231):
            segment = timeseries[:, j:j+shapelet_length]
            dist = torch.mean((segment - shapelet)**2)
            distances.append((dist.item(), j))
        
        distances = sorted(distances, key = lambda x: x[0])
        closest_match = distances[0]
        print(closest_match)

        start_idx = closest_match[1]

        padded_shapelet = torch.cat([
        torch.full((start_idx,), float('nan')),  # Padding with NaNs before the start index
        shapelet.squeeze(0),                     # The shapelet itself
        torch.full((len(timeseries.squeeze()) - (start_idx + shapelet_length),), float('nan'))  # Padding after the shapelet
        ])
        axes[i].plot(timeseries.squeeze().cpu().numpy(), label=f"Time Series", color="b")
        axes[i].plot(padded_shapelet.cpu().detach().numpy(), 
                 color="g", linewidth=2, label=f"Shapelet {i+1}")
        axes[i].legend()

    plt.tight_layout()
    plt.show()

