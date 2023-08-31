import os
import pickle
import requests
import numpy as np
import argparse
from tqdm.auto import tqdm
from tokenizers import ByteLevelBPETokenizer



def prepare(tokenizer:str, dataset:str, check_decode:bool=False):
    print(tokenizer+'-vocab.json', '\n'+tokenizer+'-merges.txt')
    tokenizer = ByteLevelBPETokenizer(tokenizer+'-vocab.json', tokenizer+'-merges.txt')
    num_paragraphs = 0
    text_path = f'data/{dataset}/input.txt'
    bin_path = f'data/{dataset}/train.bin'
    with open(text_path, 'r') as f:
        while f.readline():
            num_paragraphs += 1
    train_ids = []
    with open(text_path, 'r') as f:
        pbar = tqdm(total=num_paragraphs)
        line = f.readline()
        while line:
            train_ids.extend(tokenizer.encode(line, add_special_tokens=False).ids)
            line = f.readline()
            pbar.update()
    train_ids = np.array(train_ids, dtype=np.uint16)
    train_ids.tofile(bin_path)
    if check_decode:
        out_str = ''
        for _id in train_ids:
            out_str += tokenizer.decode([_id])
        if out_str != open(text_path, 'r').read():
            print('Decode doesnt equal input. Saving to debug.txt')
            with open('debug.txt', 'w') as f:
                f.write(out_str)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='tokyo-article')
    parser.add_argument('--tokenizer', type=str, default='data/wiki9/enwik9_tokenizer_2000')
    parser.add_argument('--check_decode', action='store_true')
    args = parser.parse_args()
    prepare(args.tokenizer, args.dataset, args.check_decode)

