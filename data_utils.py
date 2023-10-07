import os
import torch 
from torch.utils.data import Dataset
import numpy as np


class OrderedTextDataset(Dataset):
    def __init__(self, dataset:str, block_size:int, device_type:str='cpu', device=torch.device('cpu'), tokenizer:str=None) -> None:
        super().__init__()
        self.name = dataset
        self.data_dir = f'data/{dataset}'
        # train_file_name = 'train.bin'
        train_file_name = None
        if tokenizer:
            if tokenizer.startswith('gpt2'):
                tokenizer = 'gpt2'
            train_file_name = f'train-{tokenizer}.bin'
        self.train_data =  np.memmap(os.path.join(self.data_dir, train_file_name), dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.device_type = device_type
        self.device = device
    def num_tokens(self):
        return len(self.train_data)
    def __len__(self):
        return len(self.train_data) - self.block_size - 1
    def __getitem__(self, index):
        i = index
        x = torch.from_numpy((self.train_data[i:i+self.block_size]).astype(np.int64))
        y = torch.from_numpy((self.train_data[i+1:i+1+self.block_size]).astype(np.int64))
        return x,y 
        if self.device_type == 'cuda':
            # x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
            x, y = x.to(self.device), y.to(self.device)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y


