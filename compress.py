import numpy as np 
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
    
    def binsearch(self, nums):
        target = self.get_buf_val()
        left, right = 0, len(nums)
        while left < right:
            mid = (left + right) // 2
            if nums[mid] <= target:
                left = mid + 1
            else:
                right = mid
        return right

    def _binsearch(self, nums):
        target = self.get_buf_val()
        floor_int = 0
        for i in range(len(nums)):
            ceil_int = nums[i]
            if floor_int <= target < ceil_int:
                return i
        return len(nums) - 1

    def get_cum_prob_int(self, probs, window:int=2**32, offset:int=0):
        cumint = 0
        C = []
        window = window - len(probs)
        probs_int = (probs * window).astype(np.int64)
        for i,p in enumerate(probs_int):
            cumint += p.item() + 1
            C.append(cumint + offset)
        return C

    def encode_token(self, c, probs=None, step:int=-1):
        if probs is None:
            probs = self.probs
        probs = probs.reshape(-1)
        window = self.ceiling - self.floor 
        assert window <= 2**32, f'Window should be max 2**32, but is {window-2**32} over'
        if window < len(probs):
            cumprob_int = self.get_cum_prob_int(probs.astype(np.float64))
            new_ceil = cumprob_int[c]
            if c == 0:
                new_floor = 0
            else:
                new_floor = cumprob_int[c-1]
            self.floor = new_floor 
            self.ceiling = new_ceil
            self.buf.append(new_floor + np.zeros((1,), dtype=np.uint32))
        else:
            cumprob_int = self.get_cum_prob_int(probs.astype(np.float64), window, self.floor)
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

    def load_buffer(self, path='compressed.bin'):
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

    def decode_token(self, probs=None, step:int=-1):
        if probs is None:
            probs = self.probs
        probs = probs.reshape(-1)
        for k in range(2):
            window = self.ceiling - self.floor
            if window < len(probs):
                self.pop_buf_val()
                self.floor = 0 
                self.ceiling = 2**32
                continue
            cumprobs_int = self.get_cum_prob_int(probs.astype(np.float64), window=window, offset=self.floor)
            idx = self.binsearch(cumprobs_int)
            self.ceiling = cumprobs_int[idx]
            if idx > 0:
                self.floor = cumprobs_int[idx-1]
            if self.debug:
                n = len(self.decode_ceils)
                self.decode_ceils.append(self.ceiling)
                self.decode_floors.append(self.floor)
                self.decode_cumprobints.append(cumprobs_int)
                assert self.ceiling == self.encode_ceils[n], f'Should be same'
                assert self.floor == self.encode_floors[n], f'Should be same'

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

    pass
   