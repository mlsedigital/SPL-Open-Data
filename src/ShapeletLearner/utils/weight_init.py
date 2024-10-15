import numpy as np
from sklearn.cluster import KMeans

def extract_segments(timeseries_data, segment_length):
    _, time_series_length = timeseries_data.shape
    segments = []
    for ts in timeseries_data:
        num_segments = time_series_length - segment_length + 1
        for i in range(num_segments):
            segment = ts[i:i+segment_length]
            segments.append(segment)
    
    return np.array(segments)


def shapelet_initialization(timeseries_data, num_shapelets, shapelet_length):
    segments = extract_segments(timeseries_data, shapelet_length)
    kmeans = KMeans(n_clusters=num_shapelets, random_state=42)
    kmeans.fit(segments)
    return kmeans.cluster_centers_ 