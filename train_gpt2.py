import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

from gpt2 import GPT, GPTConfig


max_length = 30
num_sequences = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device=device)

def load_tokens(filenamne: str) -> torch.Tensor:
    npt = np.load(filenamne)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataloaderLite:

    def __init__(self, B: int, T: int, split: str):
        self.B = B
        self.T = T                        

        data_root = "fineweb"
        shards = os.listdir(data_root)
        shards = sorted([s for s in shards if split in s])
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards

        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_pos = self.B*self.T
        self.reset()

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_pos:self.current_pos+B*T+1]
        x = buf[:-1].view(B, T)
        y = buf[1:] .view(B, T)

        self.current_pos += B*T

        if self.current_pos + (B*T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard+1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_pos = B*T
        return x, y
    
    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_pos = B*T


import time
torch.manual_seed(42)
torch.cuda.manual_seed(42)

model = GPT(GPTConfig(vocab_size=50304))
model = torch.compile(model)
model.train()

optimizer = model.configure_optimizers(weight_decay=0.1, lr=6e-4, betas=(0.9, 0.95))

# total_batch_size = 2**19
B = 4
T = 1024
# assert total_batch_size % (B*T) == 0

# grad_accum_steps = total_batch_size // (B*T)
grad_accum_steps = 1
train_loader = DataloaderLite(B=B, T=T, split="train")
val_loader = DataloaderLite(B=B, T=T, split="val")

torch.set_float32_matmul_precision('high')

max_steps = 2_000
warmup_steps = 100
max_lr = 6e-5
min_lr = max_lr * 0.1

val_interval = 100
val_loss_steps = 20

def get_lr(it: int) -> float:
    # linear warmup for warmup_steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # if it > max_step just return min_lr
    if it > max_steps:
        return min_lr
    # in between use cosine decay
    decay_ratio = (it-warmup_steps) / (max_steps-warmup_steps)
    assert 0<= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(decay_ratio*math.pi))
    return min_lr + coeff * (max_lr-min_lr)


import time

enc = tiktoken.get_encoding('gpt2')
scaler = torch.GradScaler(device="cuda")
start = time.perf_counter()

for step in range(max_steps):
    t0 = time.perf_counter()

    if step % val_interval == 0:
        model.eval
        val_loader.reset()
        with torch.no_grad():
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()                
                logits, loss = model(x, y)    
        print(f"Val Loss: {loss.item():.4f}")

    if step > 0 and step % val_interval == 0:
        model.eval()
        num_sequences = 4
        max_length = 32
        prefix = "Hello, I am from another planet take"
        tokens = enc.encode(prefix)
        tokens = torch.tensor(tokens, dtype=torch.long)
        x = tokens.unsqueeze(0).repeat(num_sequences, 1)
        while x.size(1) < max_length:
            with torch.no_grad():
                logits, _ = model(x)        
                logits = logits[:, -1, :] # get last tokens logits
                probs = F.softmax(logits, dim=-1)
                top_k_probs, top_k_idx = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(top_k_probs, num_samples=1)
                xcol = torch.gather(top_k_idx, dim=-1, index=ix)
                x = torch.cat((x, xcol), dim=1)

            for i in range(num_sequences):
                tokens = x[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(f"sample: {i}, decoded: {decoded}")

    optimizer.zero_grad()
    model.train()
    x, y = train_loader.next_batch()
    
    logits, loss = model(x, y)    
    loss.backward()
    norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    dt = (t1-t0)*1000
    tk_per_ms = (train_loader.B * train_loader.T *grad_accum_steps)/dt
    print(f"Step: {step} | lr: {lr:.4e} | Loss: {loss.item():.4f} | time: {dt:.4f}ms | tokens/ms: {tk_per_ms:.4f} | norm: {norm:.4f}")

tot = time.perf_counter() - start
print(f"{tot:.2f}s")



import sys; sys.exit(1)
prefix = "hello, my name is L"
tokens = enc.encode(prefix)
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_sequences, 1)
x = tokens.to(device)

while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)        
        logits = logits[:, -1, :] # get last tokens logits
        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_idx = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(top_k_probs, num_samples=1)
        xcol = torch.gather(top_k_idx, dim=-1, index=ix)
        x = torch.cat((x, xcol), dim=1)

for i in range(num_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(decoded)