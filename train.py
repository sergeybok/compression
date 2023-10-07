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
from model import LMCompressorBase
from train_compress import init_model, init_seeds, format_duration
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

COMPILE = False
DTYPE = torch.float32




class TrainParams(BaseModel):
    batch_size:int=32
    lr:float=0.0004
    beta1:float=0.9
    beta2:float=0.999
    weight_decay:float=0
    grad_clip:float=1.0
    warmup:int=40
    use_scheduler:bool=True
    cycle_steps:int=1_000_000
    gradient_accumulation_steps:int=1
    window_size:int=200
    overlap_size:int=0
    vocab_size:int = 5000
    model_name:str='GPTLarge2L' # ['gpt2pretrained'  'gpt2real', 'gpt2xl','gpt2medium', 'rnnbig', 'lstmbig','lstmmedium', 'gpt2small' 'gpt2tiny', 'gpt2']
    tokenizer:str='wiki9'
    model_params: BaseModel = None
    device_type:str = 'cuda' # 'cuda'
    dataset_name:str = 'wiki9' # ['tokyo-article', 'wiki9', 'shakespeare']
    seed:int=1609
    swa_start:int=500_000
    num_steps:int=1_000_000





def train(params:TrainParams, extra_name:str=''):
    name = f'train_{params.dataset_name}_{params.model_name}_seed_{params.seed}_{params.tokenizer.replace(params.dataset_name, "")}-{params.vocab_size}-{params.window_size}_{extra_name}'
    writer = SummaryWriter(f'logs_compress/{name}')
    with open(f'logs_compress/{name}/meta.json', 'w') as f:
        json.dump({'params': str(params)}, f)
    device = torch.device(params.device_type)
    
    model:LMCompressorBase = init_model(params).to(device)
    optimizer, scheduler = model.configure_optimizer()
    swa_scheduler = SWALR(optimizer, swa_lr=0.05)
    if COMPILE:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0
    if 'pretrained' in params.model_name:
        params.dataset_name
    dataset = OrderedTextDataset(params.dataset_name, tokenizer=f'{params.tokenizer}_{params.vocab_size}', block_size=params.window_size, device_type=params.device_type, device=device)
    ctx = nullcontext() if params.device_type == 'cpu' else torch.amp.autocast(device_type=params.device_type, dtype=DTYPE)
    # ctx = nullcontext()
    swa_model = AveragedModel(model)
    def save_models():
        exp_path = f'experiments/{name}'
        model_path = f'{exp_path}/model.ckpt'
        swa_model_path = f'{exp_path}/swa_model.ckpt'
        optim_path = f'{exp_path}/adamw.ckpt'
        torch.save(model.state_dict(), model_path)
        torch.save(swa_model.state_dict(), swa_model_path)
        torch.save(optimizer.state_dict(), optim_path)


    model.train()
    model = model.to(device)
    pbar = tqdm(total=params.num_steps)
    global_step = 0
    exp_avg_loss = None
    start_time = int(time.time())
    dataloader = DataLoader(dataset=dataset, batch_size=params.batch_size, shuffle=True, num_workers=8)
    finished = False
    def save_state():
        duration = int(time.time()) - start_time
        exp_path = f'experiments/{name}'
        os.makedirs(exp_path, exist_ok=True)
        with open(f'{exp_path}/compress_meta.json', 'w') as f:
            json.dump({'duration': format_duration(duration), 'steps': global_step, 'params': params.dict()}, f)
    for epoch in range(20):
        for batch in dataloader:
            X,Y = batch
            with ctx:
                optimizer.zero_grad()
                _, loss = model.train_step(X.to(device), Y.to(device))
                loss.backward()
                if params.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), params.grad_clip)
                optimizer.step()
                if scheduler and global_step < params.swa_start:
                    scheduler.step()
            if exp_avg_loss is None:
                exp_avg_loss = loss.item()
            else:
                exp_avg_loss = (exp_avg_loss + loss.item()) / 2
            if global_step % 50 == 0:
                writer.add_scalar('train_loss_smoothed', exp_avg_loss, global_step=global_step)
                pbar.set_description(f'E={epoch} loss={round(exp_avg_loss, 3)}')
                if global_step > params.swa_start:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()
            if global_step+1 % 50_000 == 0:
                save_state()
                save_models()
            global_step += 1
            pbar.update()
            if global_step > params.num_steps:
                finished = True 
                break
        if finished:
            break
    save_state()
    save_models()



if __name__ == '__main__':

    params = TrainParams()
    params.num_steps = 2_000_001
    params.cycle_steps = 30_001

    num = 0
    for lr in [8e-5]:
        params.lr = lr
        params.grad_clip = 0 
        for b1 in [0.9]:
            print(f'running  lr={lr} b1={b1}')
            params.beta1 = b1
            train(params, f'lr={lr*100}_b1{b1}_2')
            num += 1
