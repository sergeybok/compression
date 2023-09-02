import numpy as np 
cimport numpy as np

cdef class ArithmeticCoding:
    cdef long floor, ceiling
    cdef np.ndarray probs
    cdef list buf

    def __init__(self, np.ndarray[double, ndim=1] probs=None) -> None:
        self.probs = probs 
        self.clear_buffer()

    cdef list get_cum_prob_int(self, np.ndarray[double, ndim=1] probs, long window=2**32, long offset=0):
        cdef long cumint = 0
        cdef list C = []
        window = window - len(probs)
        cdef np.ndarray[np.int64_t, ndim=1] probs_int = (probs * window).astype(np.int64)
        cdef int i
        cdef np.int64_t p
        for i, p in enumerate(probs_int):
            cumint += p + 1
            C.append(cumint + offset)
        return C
    
    cdef int binsearch(self, list nums):
        cdef long target = self.get_buf_val()
        cdef int left, right, mid
        left, right = 0, len(nums)
        while left < right:
            mid = (left + right) // 2
            if nums[mid] <= target:
                left = mid + 1
            else:
                right = mid
        return right
    
    def encode_token(self, int c, np.ndarray[double, ndim=1] probs=None):
        if probs is None:
            probs = self.probs
        probs = probs.reshape(-1)
        cdef unsigned long window = self.ceiling - self.floor 
        assert window <= 2**32, f'Window should be max 2**32, but is {window-2**32} over'
        cdef list cumprob_int
        cdef long new_ceil, new_floor
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

    def cache_buffer(self, str path='compressed.bin'):
        np.array(self.buf, dtype=self.buf[0].dtype).tofile(path)

    def load_buffer(self, str path='compressed.bin'):
        self.buf = [n for n in np.memmap(path, dtype=np.uint32, mode='r')]
    
    def clear_buffer(self):
        self.buf = [np.zeros((1,), dtype=np.uint32)]
        self.floor = 0
        self.ceiling = 2**32
        return
    
    def get_buf_val(self):
        return self.buf[0]
    
    def pop_buf_val(self):
        self.buf = self.buf[1:]
    
    def decode_token(self, np.ndarray[double, ndim=1] probs=None):
        if probs is None:
            probs = self.probs
        probs = probs.reshape(-1)
        cdef int k
        cdef unsigned long window
        cdef list cumprobs_int
        cdef int idx
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
            return idx
        raise Exception()
