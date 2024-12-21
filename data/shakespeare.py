"""
FineWeb dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb

example doc to highlight the structure of the dataset:
{
  "text": "Posted by mattsmith on 20th April 2012\nStraight from...",
  "id": "<urn:uuid:d853d453-196e-4488-a411-efc2b26c40d2>",
  "dump": "CC-MAIN-2013-20",
  "url": "http://nleastchatter.com/philliesphandom/tag/freddy-galvis/",
  "date": "2013-05-18T07:24:47Z",
  "file_path": "s3://commoncrawl/long.../path.../file.gz",
  "language": "en",
  "language_score": 0.9185474514961243,
  "token_count": 594
}
"""
import os
import argparse
import multiprocessing as mp
import numpy as np
import tiktoken
# from huggingface_hub import snapshot_download
from datasets import load_dataset
from tqdm import tqdm
import argparse
import numpy as np

# Add this near the top of the file, after imports
nprocs = mp.cpu_count()

def write_datafile(filename, toks):
    """ 
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint16
    """
    assert len(toks) < 2**31, "token count too large" # ~2.1B tokens
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520 # magic
    header[1] = 1 # version
    header[2] = len(toks) # number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
    # construct the tokens numpy array, if not already
    if not isinstance(toks, np.ndarray) or not toks.dtype == np.uint16:
        # validate that no token exceeds a uint16
        maxtok = 2**16
        assert all(0 <= t < maxtok for t in toks), "token dictionary too large for uint16"
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks
    # write to file
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())
# ------------------------------------------

parser = argparse.ArgumentParser(description="FineWeb dataset preprocessing")
parser.add_argument("-v", "--version", type=str, default="10B", help="Which version of fineweb to use 10B|100B")
parser.add_argument("--train_split", type=float, default=0.9, help="Fraction of data to use for training")
args = parser.parse_args()

# FineWeb has a few possible subsamples available
local_dir = "shakespeare"

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = '/mnt/rd/dataset'
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
fw = load_dataset("karpathy/tiny_shakespeare",split="train")

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

# Collect all tokens first
all_tokens = []
with mp.Pool(nprocs) as pool:
    for tokens in tqdm(pool.imap(tokenize, fw, chunksize=16), desc="Tokenizing"):
        all_tokens.append(tokens)

# Concatenate all tokens
all_tokens = np.concatenate(all_tokens)
n = len(all_tokens)

# Split into train/val
train_tokens = all_tokens[:int(n*args.train_split)]
val_tokens = all_tokens[int(n*args.train_split):]

# Write train and val files
write_datafile(os.path.join(DATA_CACHE_DIR, "shakespeare_train_000000.bin"), train_tokens)
write_datafile(os.path.join(DATA_CACHE_DIR, "shakespeare_val_000000.bin"), val_tokens)
