from datasets import load_dataset
import tiktoken
import os
import numpy as np
from tqdm.auto import tqdm
import torch

def download_dataset() : 
    dataset = load_dataset("roneneldan/TinyStories")
    return dataset

def get_tokenizer() : 
    tokenizer = tiktoken.get_encoding("gpt2")
    return tokenizer

def process(data,tokenizer) :
    ids = tokenizer.encode_ordinary(data["text"])
    out = {'ids':ids , "len" : len(ids)}
    return out

def Build_Dataset(dataset) : 
    if not os.path.exists('train.bin'):
        tokenized = dataset.map(
        process,
        remove_columns=["text"],
        desc="Tokenizing text",
        num_proc=8,
        )
        for split , dset in tokenized.items() :
            arr_len = np.sum(dset["len"],dtype=np.int64)
            filename = f"{split}.txt"
            dtype = np.uint16
            arr = np.memmap(filename, dtype=dtype, mode="w+",shape=(arr_len,))
            total_batches = 1024
            idx = 0
            for batch_idx in tqdm(range(total_batches) , desc=f"Writing {filename}") :
                batch = dset.shard(num_shards=total_batches,index=batch_idx,contiguous=True)
                arr_batch = np.concatenate(batch["ids"])
                ## write mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()
        

def get_batch(split,block_size,batch_size,device) : 
    if split == "train":
        data = np.memap("train.bin",dtype=np.uint16,mode='r')
    else : 
        data = np.memap("validation.bin",dtype=np.uint16,mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device == "cuda" : 
        x,y = x.pin_memory().to(device,non_blocking=True) , y.pin_memory().to(device,non_blocking=True)
    else : 
        x,y = x.to(device),y.to(device)
    return x, y