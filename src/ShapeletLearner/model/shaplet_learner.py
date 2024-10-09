import torch
import torch.nn as nn
import torch.nn.functional as F


class ShapletLearner(nn.Module):
    def __init__(self, num_shapelet, len_shapelet, input_size, num_classes, alpha_precision, kmeans_centroids):
        super().__init__()

        assert input_size % len_shapelet  == 0, "The input size may not be compatiable with the Timeseries size" 
        self.L = num_shapelet ## Using the paper's terminalogy
        self.K = len_shapelet

        self.Q = input_size
        self.I = num_classes

        self.alpha = alpha_precision

        if kmeans_centroids is not None:
            assert kmeans_centroids.shape == (num_shapelet, len_shapelet), \
                "KMeans centroids must have the shape (num_shapelets, len_shapelet)."
            self.shapelets = nn.ParameterList(
                [nn.Parameter(torch.tensor(centroid, dtype=torch.float32)) for centroid in kmeans_centroids]
            )
        else:
            self.shapelets = nn.ParameterList(nn.Parameter(torch.randn(self.K)) for _ in range(num_shapelet))
        
       
        self.fc1 = nn.Linear(self.L, 2)

        self.loss_fn = torch.nn.CrossEntropyLoss() ## This is what they used not sure why not BCE with logit but eh

    def _compute_shapelet_dist(self, ts):
        shapelet_distances = [] 
        for shapelet in self.shapelets:
            num_segments = self.Q - self.K + 1
            distances = []
            for j in range(num_segments):
                segment = ts[:, j:j+self.K] 
                segment = segment
                dist = torch.mean((segment - shapelet)**2, dim = 1)
                if torch.isnan(dist).any():
                    print("NaN detected in distance calculation")
                distances.append(dist)
            distances = torch.stack(distances, dim = 1)

            hard_min, _ = torch.min(distances, dim=1)
            if torch.isnan(hard_min).any():
                print("NaN detected in min operation") 
            shapelet_distances.append(hard_min)

        result =  torch.stack(shapelet_distances, dim = 1) 
        if torch.isnan(result).any():
            print("NaN detected in final Shapelet distance")
        return torch.stack(shapelet_distances, dim = 1) 

    def forward(self, ts):
        ts = ts.float()
        shaplet_distances = self._compute_shapelet_dist(ts)
        output = self.fc1(shaplet_distances)
        return output
    
    def loss(self, pred, labels):
        return self.loss_fn(pred, labels)