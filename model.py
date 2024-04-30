import torch
import torch.nn as nn
import torch.nn.functional as F


# hyperparameters
batch_size = 32
block_size = 8
iterations = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# open training data
with open('../data/input.txt', 'r', encoding='utf-8') as f:
	text = f.read()

# get vocab
chars = sorted(list(set(text)))
vocab_size = len(chars)

# tokenization
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda ixs: ''.join([itos[i] for i in ixs])

# encode data
data = torch.tensor(encode(text), dtype=torch.long)
# split up data
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# load data
def get_batch(split):
	data = train_data if split == 'train' else val_data
	ix = torch.randint(len(data) - block_size, (batch_size,))
	x = torch.stack([data[i:i+block_size] for i in ix])
	y = torch.stack([data[i+1:i+block_size+1] for i in ix])
	return x,y

