import torch
from tqdm.auto import tqdm 
from arithmetic_coding import ArithmeticCoding
# from compress import ArithmeticCoding


FN = 'data/shakespeare/input.txt'
# FN = 'data/tokyo-article/input.txt'


with open(FN, 'r') as f:
    txt = f.read() 
alphabet = sorted(list(set(txt)))
print(len(alphabet))
P = {k:0 for k in alphabet}
for c in txt:
    P[c] += 1 
for c,n in P.items():
    P[c] = n / len(txt)
print(P)


probs = []
for c in alphabet:
    probs.append(P[c])

probs = torch.tensor(probs)

alpha_to_index = { alphabet[i]:i for i in range(len(alphabet)) }

def print_sorted():
    print('Alphabet Sorted ')
    i = 0
    while i < len(alphabet):
        s = ', '.join(alphabet[i:i+5])
        print(i, s)
        i += 5


print_sorted()


compressor = ArithmeticCoding(probs.double().numpy())
i = 0
for c in tqdm(txt):
    idx = alpha_to_index[c]
    if i == 432:
        x = 1
    compressor.encode_token(idx)
    i += 1
    pass

compressor.cache_buffer('debug.bin')
compressor.clear_buffer()

compressor.load_buffer('debug.bin')

out_str = ''
for i in tqdm(range(len(txt))):
    idx = compressor.decode_token()
    out_str += alphabet[idx]
    pass

with open('debug.txt', 'w') as f:
    f.write(out_str)


assert out_str == txt
