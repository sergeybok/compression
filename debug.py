from compress import ArithmeticCoding
import torch
from tqdm.auto import tqdm 



with open('article.txt', 'r') as f:
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


compressor = ArithmeticCoding(probs, debug=True)

for c in tqdm(txt):
    idx = alpha_to_index[c]
    compressor.encode_token(idx)
    pass

compressor.cache_buffer('article.pkl')


compressor.clear_buffer()
compressor.load_buffer('article.pkl')

out_str = ''
for i in range(len(txt)):
    idx = compressor.decode_token()
    out_str += alphabet[idx]
    pass

with open('aritcle_hat.txt', 'w') as f:
    f.write(out_str)


assert out_str == txt