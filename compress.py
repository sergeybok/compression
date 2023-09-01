from train import LSTMModelBeefy, WikiDataset
from tokenizers import ByteLevelBPETokenizer
import numpy as np 
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm 


    

class ArithmeticCoding:
    def __init__(self, probs=None, debug=False) -> None:
        self.probs = probs 
        self.clear_buffer()
        self.debug = debug
        if debug:
            self.encode_floors = []
            self.encode_ceils = []
            self.decode_floors = []
            self.decode_ceils = []
            self.encode_cumprobints = []
            self.decode_cumprobints = []
            self._window = 0
    def binsearch(self, nums) -> int:
        target = self.get_buf_val()
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return right
    def binsearch(self, nums) -> int:
        target = self.get_buf_val()
        floor_int = self.floor
        for i in range(0, len(nums)):
            if floor_int <= target < nums[i]:
                return i 
            floor_int = nums[i]
        return len(nums) - 1

    def get_cum_prob_int(self, probs, window:int=2**32, offset:int=0):
        probs = probs.to(torch.float64)
        cumint = 0
        C = []
        window = window - len(probs)
        probs_int = probs * window + 0.5
        probs_int = probs_int.to(torch.int64)
        for i,p in enumerate(probs_int):
            cumint += p.item() + 1
            C.append(cumint + offset)
        return C

    def encode_token(self, c, probs=None):
        if probs is None:
            probs = self.probs
        probs = probs.view(-1)
        window = self.ceiling - self.floor 
        assert window <= 2**32, f'Window should be max 2**32, but is {window-2**32} over'
        if window < len(probs):
            cumprob_int = self.get_cum_prob_int(probs.type(torch.float64))
            new_ceil = cumprob_int[c]
            if c == 0:
                new_floor = 0
            else:
                new_floor = cumprob_int[c-1]
            self.floor = new_floor 
            self.ceiling = new_ceil
            self.buf.append(new_floor + np.zeros((1,), dtype=np.uint32))
        else:
            cumprob_int = self.get_cum_prob_int(probs.type(torch.float64), window, self.floor)
            new_ceil = cumprob_int[c]
            if c == 0:
                new_floor = self.floor
            else:
                new_floor = cumprob_int[c-1]
            self.buf[-1][0] = new_floor
            self.floor = new_floor
            self.ceiling = new_ceil 
        if self.debug:
            self.encode_floors.append(self.floor)
            self.encode_ceils.append(self.ceiling)
            self.encode_cumprobints.append(cumprob_int)

    def cache_buffer(self, path='compressed.bin'):
        np.array(self.buf, dtype=self.buf[0].dtype).tofile(path)

    def load_buffer(self, path='buf.npy'):
        self.buf = np.memmap(path, dtype=np.uint32, mode='r')

    def clear_buffer(self):
        self.buf = [np.zeros((1,), dtype=np.uint32)]
        self.floor = 0
        self.ceiling = 2**32
        return
    def get_buf_val(self):
        return self.buf[0]
    def pop_buf_val(self):
        self.buf = self.buf[1:]

    def decode_token(self, probs=None):
        if probs is None:
            probs = self.probs
        probs = probs.view(-1)
        for k in range(2):
            window = self.ceiling - self.floor
            if window < len(probs):
                self.pop_buf_val()
                self.floor = 0 
                self.ceiling = 2**32
                continue
            cumprobs_int = self.get_cum_prob_int(probs.type(torch.float64), window=window, offset=self.floor)
            idx = self.binsearch(cumprobs_int)
            self.ceiling = cumprobs_int[idx]
            if idx > 0:
                self.floor = cumprobs_int[idx-1]
            if self.debug:
                self.decode_ceils.append(self.ceiling)
                self.decode_floors.append(self.floor)
                self.decode_cumprobints.append(cumprobs_int)
            return idx
        raise Exception()
 

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--run', type=str, default='logs/seed_631', help='Run path eg logs/seed_631')
    parser.add_argument('--encode', action='store_true', help='Encode ')
    parser.add_argument('--limit', type=int, default=-1, help='')
    parser.add_argument('--save_path', type=str, default='wiki9_compressed.pkl', help='')
    return parser.parse_args()



if __name__ == '__main__':
    from train import Params
    import json 
    args = parse_args()
    path = args.run #f'logs/seed_631'
    chunk_size = 500
    
    with open(f'{path}/meta.json', 'r') as f:
        p = json.load(f)['params'].split(' ')
        pdict = {}
        for t in p:
            if '=' in t:
                k,v = t.split('=')
                pdict[k] = v
    params = Params.parse_obj(pdict)

    tokenizer = ByteLevelBPETokenizer('enwik9_tokenizer_2000-vocab.json','enwik9_tokenizer_2000-merges.txt')
    model = LSTMModelBeefy(params.vocab_size, params.embedding_dim, params.hidden_dim, params.n_layers, params.use_ortho, params.use_ghazi_init, params.use_xavier, input_bias=params.input_bias)
    softmax = torch.nn.Softmax(dim=2)
    dataset = WikiDataset(chunk_size, tokenizer, limit=args.limit)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.load_state_dict(torch.load(f'{path}/lstm_model.ckpt'))
    compressor = ArithmeticCoding()
    num_el = 0
    numsteps = 0
    with torch.no_grad():
        pbar = tqdm(total=len(dataloader))
        hidden = model.init_hidden(1)
        for bidx, (batch, tgt, slen) in enumerate(dataloader):
            outputs, hidden = model(batch, hidden)
            probs = softmax(outputs)
            # print('insize', batch.size())
            for t in range(batch.size(1)):
                try:
                    compressor.encode_token(tgt[0,t], probs[0, t])
                except Exception as e:
                    print(tgt[0,t].size(), probs[0, t].size())
                    raise(e)
                numsteps += 1
            num_el += slen
            pbar.set_description(f'len(S)={num_el} len(buf)={len(compressor.buf)}')
            pbar.update()
        compressor.cache_buffer(args.save_path)
    compressor.clear_buffer()
    compressor.load_buffer(args.save_path)
    
    with torch.no_grad():
        pbar = tqdm(total=numsteps)
        hidden = model.init_hidden(1)
        out_str = ''
        prev_token = dataset.first_token()
        for i in range(numsteps):
            batch = torch.tensor(prev_token, dtype=torch.long).view(1,1)
            outputs, hidden = model(batch, hidden)
            print('outputs shape', outputs.size())
            probs = softmax(outputs)
            print('probs shape', probs.size())
            prev_token = compressor.decode_token(probs)
            out_str += tokenizer.decode(prev_token)
            pbar.update()
    with open('debug.txt', 'w') as f:
        f.write(out_str)
            





   