import os
import torch 
from torch.utils.data import Dataset
import numpy as np


class OrderedTextDataset(Dataset):
    def __init__(self, dataset:str, block_size:int, device_type:str='cpu', device=torch.device('cpu')) -> None:
        super().__init__()
        self.name = dataset
        self.data_dir = f'data/{dataset}'
        self.train_data =  np.memmap(os.path.join(self.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.device_type = device_type
        self.device = device
    def __len__(self):
        return len(self.train_data) - self.block_size - 1
    def __getitem__(self, index):
        i = index
        x = torch.from_numpy((self.train_data[i:i+self.block_size]).astype(np.int64))
        y = torch.from_numpy((self.train_data[i+1:i+1+self.block_size]).astype(np.int64))
        if self.device_type == 'cuda':
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y


