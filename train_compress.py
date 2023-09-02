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
import os 
from contextlib import nullcontext

# from compress import ArithmeticCoding
from arithmetic_coding import ArithmeticCoding

from data_utils import OrderedTextDataset
from model import LMCompressorBase


COMPILE = False
DTYPE = torch.float32

class LSTMParams(BaseModel):
    lr:float= 0.0004
    beta1:float=0.9
    beta2:float=0.95
    warmup:int=40_000
    cycle_steps:int=40_000
    use_ortho:bool=False
    use_ghazi_init:bool=False
    use_xavier:bool=False
    beefy:bool=True
    vocab_size:int = 2000
    embedding_dim:int = 200
    hidden_dim:int = 400
    n_layers:int = 1
    chunk_size:int = 600
    use_scheduler:bool=True
    input_bias:bool=True
    # embedding_path:str='embeddings.pt'
    embedding_path:str=None


class TrainParams(BaseModel):
    lr:float=0.0004
    beta1:float=0.5
    beta2:float=0.999 
    weight_decay:float=1e-5
    grad_clip:float=2.0
    warmup:int=100
    use_scheduler:bool=True
    cycle_steps:int=-1
    gradient_accumulation_steps:int=1
    window_size:int=100
    overlap_size:int=0
    vocab_size:int = 2000
    model_name:str='rnnbig' # [ 'rnnbig', 'lstmbig','lstmmedium', 'gpt2small' 'gpt2tiny', 'gpt2']
    model_params: BaseModel = None
    device_type:str = 'cpu'
    dataset_name:str = 'shakespeare' # ['tokyo-article', 'wiki9', 'shakespeare']
    seed:int=1584





def init_seeds(seed:int):
    torch.manual_seed(seed)
    np.random.seed(seed)

def init_model(params:TrainParams) -> LMCompressorBase:
    init_seeds(params.seed)
    print(f"Initializing a new model {params.model_name} from scratch")
    # determine the vocab size we'll use for from-scratch training
    if params.model_name == 'gpt2small':
        from model import GPTSmall
        model = GPTSmall(params)
        return model
    elif params.model_name == 'lstmbig':
        from model import LSTMBig
        model = LSTMBig(params)
        return model
    elif params.model_name == 'lstmmedium':
        from model import LSTMMedium
        model = LSTMMedium(params)
        return model
    elif params.model_name == 'rnnbig':
        from model import RNNBig
        model = RNNBig(params)
        return model
    elif params.model_name == 'lstm':
        raise NotImplementedError('TODO')
    else:
        raise NotImplementedError(f'{params.model_name} not supported. only gpt2small and lstm')


def train_compress(params:TrainParams):
    name = f'compress_{params.dataset_name}_{params.model_name}_seed_{params.seed}'
    writer = SummaryWriter(f'logs_compress/{name}')
    with open(f'logs_compress/{name}/meta.json', 'w') as f:
        json.dump({'params': str(params)}, f)
    model:LMCompressorBase = init_model(params)
    optimizer, scheduler = model.configure_optimizer()
    device = torch.device(params.device_type)
    if COMPILE:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0
    dataset = OrderedTextDataset(params.dataset_name, block_size=params.window_size, device_type=params.device_type, device=device)
    ctx = nullcontext() if params.device_type == 'cpu' else torch.amp.autocast(device_type=params.device_type, dtype=DTYPE)

    model.train()
    total_steps = math.ceil(len(dataset)/ model.get_step_size())
    pbar = tqdm(total=total_steps)
    it = 0
    global_step = 0
    exp_avg_loss = None
    total_loss = 0
    while it < len(dataset):
        X,Y = dataset[it]
        with ctx:
            optimizer.zero_grad()
            loss, masked_loss = model.compress_step(X.unsqueeze(0),Y.unsqueeze(0))
            loss.backward()
            if params.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), params.grad_clip)
            optimizer.step()
            if scheduler:
                scheduler.step()
            total_loss += masked_loss.item() * model.get_step_size()
        if exp_avg_loss is None:
            exp_avg_loss = masked_loss.item()
        else:
            exp_avg_loss = (exp_avg_loss + masked_loss.item()) / 2
        if global_step % 50 == 0:
            writer.add_scalar('loss_smoothed', exp_avg_loss, global_step=it)
            pbar.set_description(f'loss={round(exp_avg_loss, 3)}')
        global_step += 1
        it += model.get_step_size()
        pbar.update()

    exp_path = f'experiments/{params.dataset_name}_{params.model_name}_seed_{params.seed}'
    os.makedirs(exp_path, exist_ok=True)
    with open(f'{exp_path}/compress_meta.json', 'w') as f:
        json.dump({'params': params.dict(), 'string_begin': dataset[0][0].detach().cpu().numpy().tolist()}, f)
    model.compressor.cache_buffer(f'{exp_path}/compressed_string.bin')




if __name__ == '__main__':

    params = TrainParams()
    train_compress(params)
