
from torch.utils.data import Dataset

class TSDataset(Dataset):
    def __init__(self, ts_data, labels):
        self.time_series = ts_data
        self.labels = labels
    
    def __len__(self):
        return len(self.time_series)
    
    def __getitem__(self, idx):
        return self.time_series[idx], self.labels[idx]