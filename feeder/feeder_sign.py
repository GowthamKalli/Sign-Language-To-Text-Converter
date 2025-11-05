import numpy as np
import torch
from torch.utils.data import Dataset

class FeederSign(Dataset):
    def __init__(self, data_path, label_path=None):
        self.data_path = data_path
        self.label_path = label_path
        self.load_data()

    def load_data(self):
        data = np.load(self.data_path, allow_pickle=True)
        self.data = data['x'] if 'x' in data else data
        if self.label_path:
            self.labels = np.load(self.label_path)
        else:
            self.labels = np.zeros(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        skeleton = self.data[index]
        label = int(self.labels[index])
        return torch.tensor(skeleton, dtype=torch.float32), label
