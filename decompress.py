import torch, math
import argparse
import torch.nn as nn
import torch.optim as optim
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm 
from torch.utils.tensorboard import SummaryWriter
from pydantic import BaseModel
import json
import numpy as np 
from typing import Optional
import os, time
from contextlib import nullcontext

from data_utils import OrderedTextDataset
from model import LMCompressorBase
from model import GPTCompressModel, load_gpt_model
from utils.utils import TimeIt
from compress import TrainParams, format_duration, init_seeds



COMPILE = True
DTYPE = torch.float16
DEBUG = True

class DecoderState:
    def __init__(self, batch_size:int, overlap_size:int, window_size:int, bufs:list, device) -> None:
        self.batch_size = batch_size
        self.overlap_size = overlap_size 
        self.window_size = window_size
        self.device = device
        self.window_step = 1
        if bufs is None:
            self.bufs = [[]]*batch_size
        else:
            self.bufs = bufs
    def append(self, tokens):
        for l, c in zip(self.bufs, tokens):
            l.append(c[0])
    def __str__(self) -> str:
        return ''.join([''.join(s) for s in self.bufs])
    def __len__(self):
        o = 0 
        for l in self.bufs:
            o += len(l)
        return o
    def get_batch(self):
        L = []
        for i in range(self.batch_size):
            tmp = self.bufs[i][-self.overlap_size-self.window_step:]
            L.append(torch.tensor(tmp))
        X = torch.stack(L)
        self.window_step += 1 
        if self.window_step + self.overlap_size >= self.window_size:
            self.window_step = 1
        if self.device.type.startswith('cpu'):
            X = X.to(self.device)
        else:
            X = X.pin_memory().to(self.device, non_blocking=True)
        return X

    def get_flattened(self):
        o = []
        for l in self.bufs:
            o.extend(l)
        return o



def decompress(experiment_path:str):
    with open(f'{experiment_path}/compress_meta.json') as f:
        metadata = json.load(f)
        params = TrainParams.parse_obj(metadata['params'])
    print(f'Trainining? {params.train}')
    init_seeds(params.seed)
    model:LMCompressorBase = GPTCompressModel(params)
    device = torch.device(params.device_type)
    load_gpt_model(model, params.model_path, device)
    model = model.to(device)
    if params.train:
        experiment_name = f'decompress_{params.dataset_name}_l{params.n_layers}_h{params.n_heads}_v{params.vocab_size//1000}K'
        writer = SummaryWriter(f'logs_train_compress/{experiment_name}')
    if COMPILE:
        print("compiling the model... (takes a ~minute)")
        # unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0
    print('loading compressed string..')
    model.load_arithmetic_coding_state(f'{experiment_path}/compressed_string.bin')
    ctx = nullcontext() if params.device_type == 'cpu' else torch.amp.autocast(device_type=params.device_type, dtype=DTYPE)
    if params.train:
        model.train()
    else:
        model.eval()
    global_step = 0
    exp_avg_loss = None
    total_loss = 0
    start_time = int(time.time())
    if params.train:
        optimizer, scheduler = model.configure_optimizer()
    else:
        optimizer, scheduler = None, None

    total_steps = math.ceil(metadata['num_tokens'] / params.batch_size)
    total_num_tokens = metadata['num_tokens'] 
    batch_cycle = metadata['batch_cycle']
    pbar = tqdm(total=total_steps)

    decode_state = DecoderState(params.batch_size, overlap_size=params.overlap_size,
                                window_size=params.window_size, bufs=metadata['string_begin'],
                                device=device)
    debug_dict = {}
    if DEBUG:
        debug_dict['probs'] = []
        encode_debug_dict = metadata['debug_dict']

    X = decode_state.get_batch()
    cur_train = False
    batch_it = 0
    while len(decode_state) < total_num_tokens:
        with ctx:
            cur_train = X.size(1) == params.window_size and params.train
            sub_ctx = torch.no_grad() if cur_train else nullcontext()
            with sub_ctx:
                tokens, loss, probs = model.decompress_step(X, train=cur_train, first_step=global_step==0)
            decode_state.append(tokens.detach().cpu().numpy().tolist())
            X = decode_state.get_batch()
            if cur_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
            if DEBUG:
                debug_dict['probs'].append(probs.detach().cpu().numpy().tolist())


        global_step += 1
        if global_step % params.batch_size == 0:
            batch_it += 1
        if exp_avg_loss is None:
            if loss is not None:
                exp_avg_loss = loss.item()
        else:
            exp_avg_loss = exp_avg_loss*0.8 + 0.2*loss.item()
        if global_step % 50 == 0:
            if params.train:
                pbar.set_description(f'loss={round(exp_avg_loss, 3)}')
                writer.add_scalar('loss_smoothed', exp_avg_loss, global_step=global_step)
        pbar.update()
    duration = int(time.time()) - start_time
    # save the decoder state 
    idxs = decode_state.get_flattened()
    with open(f'{experiment_path}/reconstructed.json', 'w') as f:
        json.dump({'reconstructed': idxs}, f)

    x = 1

    





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--path', type=str, default='experiments_compress/shakespeare_l8_h6_v12K')
    parser.add_argument('--path', type=str, default='experiments_compress/tokyo-article_l8_h6_v12K')
    args = parser.parse_args()
    decompress(args.path)




