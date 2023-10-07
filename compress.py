import torch, math
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
from model import LMCompressorBase, GPTCompressModel, load_gpt_model
from utils.utils import TimeIt, format_duration, human_format_bytes


COMPILE = True
DTYPE = torch.float32
DEBUG = True


class TrainParams(BaseModel):
    batch_size:int=1
    lr:float=4e-4
    beta1:float=0.9
    beta2:float=0.99
    weight_decay:float=1e-5
    grad_clip:float=2.0
    warmup:int=10
    use_scheduler:bool=False
    cycle_steps:int=-1
    gradient_accumulation_steps:int=1
    window_size:int=200
    overlap_size:int=50
    vocab_size:int = 2000
    model_name:str='gpt2medium' # ['gpt2pretrained'  'gpt2real', 'gpt2xl','gpt2medium', 'rnnbig', 'lstmbig','lstmmedium', 'gpt2small' 'gpt2tiny', 'gpt2']
    tokenizer:str='wiki9'
    model_params: BaseModel = None
    model_path:str=None
    device_type:str = 'cpu'
    dataset_name:str = 'tokyo-article' # ['tokyo-article', 'wiki9', 'shakespeare']
    seed:int=1597
    n_layers:int=6
    n_heads:int=6
    train:bool=False


def init_seeds(seed:int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def compress(model_path:str, n_heads:int, n_layers:int, vocab_size:int, train:bool, device_type:str='cuda'):
    params = TrainParams(
        batch_size=1,
        train=train,
        model_path=model_path,
        n_heads=n_heads, n_layers=n_layers, 
        device_type=device_type, window_size=256, 
        overlap_size=255, vocab_size=vocab_size)
    print(f'Trainining? {train}')
    init_seeds(params.seed)
    model:LMCompressorBase = GPTCompressModel(params)
    device = torch.device(params.device_type)
    load_gpt_model(model, model_path, device)
    # ##
    # params.window_size = 256 // 2
    # params.overlap_size = params.window_size - 1

    model = model.to(device)
    experiment_name = f'{params.dataset_name}_l{n_layers}_h{n_heads}_v{vocab_size//1000}K'
    if train:
        writer = SummaryWriter(f'logs_train_compress/{experiment_name}')
    if COMPILE:
        print("compiling the model... (takes a ~minute)")
        # unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0
    dataset = OrderedTextDataset(params.dataset_name, tokenizer=f'{params.tokenizer}_{params.vocab_size}', block_size=params.window_size, device_type=params.device_type, device=device)
    ctx = nullcontext() if params.device_type == 'cpu' else torch.amp.autocast(device_type=params.device_type, dtype=DTYPE)
    model = model.eval()
    it = 0
    global_step = 0
    exp_avg_loss = None
    total_loss = 0
    start_time = int(time.time())
    if train:
        optimizer, scheduler = model.configure_optimizer()
    else:
        optimizer, scheduler = None, None

    total_steps = math.ceil(len(dataset) / (params.batch_size*model.get_step_size()))
    pbar = tqdm(total=total_steps)
    batch_cycle = math.ceil(len(dataset) / params.batch_size)
    debug_dict = {}
    if DEBUG:
        debug_dict['probs'] = []
    
    def get_batch(it:int):
        X,Y = [], []
        for i in range(params.batch_size):
            x,y = dataset[i*batch_cycle + it]
            X.append(x)
            Y.append(y)
        X = torch.stack(X)
        Y = torch.stack(Y)
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            X, Y = X.pin_memory().to(device, non_blocking=True), Y.pin_memory().to(device, non_blocking=True)
        else:
            X, Y = X.to(device), Y.to(device)
        return X, Y
    train_ctx = torch.no_grad() if params.train else nullcontext()
    with train_ctx:
        X,Y = get_batch(it)
        while it < batch_cycle:
            last_step = False
            if it+params.window_size+model.get_step_size() >= batch_cycle:
                last_step = True
                last_overlap = it+params.window_size+model.get_step_size() - batch_cycle
            with ctx:
                loss, masked_loss, probs = model.compress_step(X, Y, last_step=last_step)
                it += model.get_step_size()
                if not last_step:
                    X,Y = get_batch(it)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
            if DEBUG:
                debug_dict['probs'].append(probs[...,:100].detach().cpu().numpy().tolist())
            global_step += 1
            if exp_avg_loss is None:
                exp_avg_loss = masked_loss.item()
            else:
                exp_avg_loss = exp_avg_loss*0.8 + 0.2*masked_loss.item()
            total_loss += masked_loss.item()
            if global_step % 50 == 0:
                pbar.set_description(f'loss={round(exp_avg_loss, 3)}')
                if train:
                    writer.add_scalar('loss_smoothed', exp_avg_loss, global_step=it)
            pbar.update()
    duration = int(time.time()) - start_time
    exp_path = f'experiments_compress/{experiment_name}'
    os.makedirs(exp_path, exist_ok=True)
    string_begin = get_batch(0)[0].detach().cpu()[:,:params.overlap_size+1].numpy()
    print(f'Saving experiment to {exp_path}')
    with open(f'{exp_path}/compress_meta.json', 'w') as f:
        json.dump({'total_loss':total_loss, 'total_steps':global_step, 'batch_cycle': batch_cycle, 
                   'duration': format_duration(duration), 'params': params.dict(), 
                   'num_tokens': dataset.num_tokens(),
                   'end_overlap': last_overlap,
                   'debug_dict': debug_dict,
                   'string_begin_shape': list(string_begin.shape),
                   'string_begin': string_begin.tolist()}, f)
    model.compressor.save_codec_state(f'{exp_path}/compressed_string.bin')



if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/home/sergey/data/repos/nanoGPT/experiments/wiki9-12k-h6-l8/ckpt.pt')
    parser.add_argument('--heads', type=int, default=6)
    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--vocab_size', type=int, default=12_000)
    parser.add_argument('--device_type', type=str, default='cpu')
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    compress(args.model_path, args.heads, args.layers, args.vocab_size, args.train, device_type=args.device_type)

