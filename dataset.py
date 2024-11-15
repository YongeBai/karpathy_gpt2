import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

local_dir = "fineweb"
shard_size = int(14e3)

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
enc = tiktoken.get_encoding('gpt2')
eos = enc._special_tokens['<|endoftext|>']

fw = load_dataset("HuggingFaceFW/fineweb-edu-llama3-annotations", split="train")

def tokenizer(string: str) -> np.ndarray:
    tokens = [eos]
    tokens.extend(enc.encode_ordinary(string))
    tokens_np = np.array(tokens)
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename: str, tokens_np: np.ndarray):
    np.save(filename, tokens_np)


def save_filtered():
    shard_count = 0
    token_count = 0
    current = []
    for item in tqdm(fw):
        if item["score"] >= 4:
            tokens = tokenizer(item["text"])
            current.append(tokens)
            token_count += len(tokens)
            if token_count >= shard_size:
                shard_file = os.path.join(DATA_CACHE_DIR, f"shard_train_{shard_count:03d}.npy")
                combined_shard = np.concatenate(current)
                write_datafile(shard_file, combined_shard)
                shard_count += 1
                current = []
                token_count = 0
    if current:
        shard_file = os.path.join(DATA_CACHE_DIR, f"shard_train_{shard_count:03d}.npy")
        combined_shard = np.concatenate(current)
        write_datafile(shard_file, combined_shard)

save_filtered()        