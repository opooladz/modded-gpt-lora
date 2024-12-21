import os
import argparse
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

def write_datafile(filename, toks):
    assert len(toks) < 2**31, "token count too large"
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520 # magic
    header[1] = 1 # version
    header[2] = len(toks)
    
    if not isinstance(toks, np.ndarray) or not toks.dtype == np.uint16:
        maxtok = 2**16
        assert all(0 <= t < maxtok for t in toks), "token dictionary too large for uint16"
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks
        
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())

def tokenize(text):
    tokens = [eot]
    if isinstance(text, dict):
        tokens.extend(enc.encode_ordinary(text["text"]))
    else:
        tokens.extend(enc.encode_ordinary(text))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all()
    return tokens_np.astype(np.uint16)

# New function to handle TinyStories format
def tokenize_tinystories(item):
    return tokenize(item['text'])

parser = argparse.ArgumentParser(description="Shakespeare + TinyStories dataset preprocessing")
parser.add_argument("--shard_size", type=int, default=10**7, help="Size of each shard in tokens")
args = parser.parse_args()

DATA_CACHE_DIR = '/mnt/rd/dataset'
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

# Load datasets
print("Loading datasets...")
shakespeare = load_dataset("karpathy/tiny_shakespeare", split="train")
tinystories = load_dataset("roneneldan/TinyStories", split="train")
tinystories_val = load_dataset("roneneldan/TinyStories", split="validation")

nprocs = max(1, os.cpu_count() - 2)

# Process training data
print("Processing training data...")
with mp.Pool(nprocs) as pool:
    shard_index = 0
    all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    
    # Process Shakespeare
    for tokens in pool.imap(tokenize, shakespeare, chunksize=16):
        if token_count + len(tokens) < args.shard_size:
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=args.shard_size, desc=f"Training shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            remainder = args.shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(os.path.join(DATA_CACHE_DIR, f"combined_train_{shard_index:06d}.bin"), all_tokens_np)
            shard_index += 1
            progress_bar = None
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder
    
    # Process TinyStories train
    for tokens in pool.imap(tokenize_tinystories, tinystories, chunksize=16):
        if token_count + len(tokens) < args.shard_size:
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=args.shard_size, desc=f"Training shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            remainder = args.shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(os.path.join(DATA_CACHE_DIR, f"combined_train_{shard_index:06d}.bin"), all_tokens_np)
            shard_index += 1
            progress_bar = None
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # Write final training shard if needed
    if token_count > 0:
        write_datafile(os.path.join(DATA_CACHE_DIR, f"combined_train_{shard_index:06d}.bin"), all_tokens_np[:token_count])

# Process validation data similarly
print("Processing validation data...")
with mp.Pool(nprocs) as pool:
    shard_index = 0
    all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    
    for tokens in pool.imap(tokenize_tinystories, tinystories_val, chunksize=16):
        if token_count + len(tokens) < args.shard_size:
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=args.shard_size, desc=f"Validation shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            remainder = args.shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(os.path.join(DATA_CACHE_DIR, f"combined_val_{shard_index:06d}.bin"), all_tokens_np)
            shard_index += 1
            progress_bar = None
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # Write final validation shard if needed
    if token_count > 0:
        write_datafile(os.path.join(DATA_CACHE_DIR, f"combined_val_{shard_index:06d}.bin"), all_tokens_np[:token_count])
