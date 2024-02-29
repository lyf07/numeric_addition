import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embed(x)
    

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.embed_dim % config.n_heads == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.embed_dim, config.embed_dim)
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_heads

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


def position_embedding(embed_dim, max_len):
    '''
    @returns ret: (max_len, embed_dim)
    '''
    ret = torch.zeros(1, max_len, embed_dim)
    for pos in range(max_len):
        for j in range(0, embed_dim, 2):
            ret[:, pos, j] = math.sin(pos / pow(10000, j / embed_dim))
            if j + 1 <= embed_dim - 1:
                ret[:, pos, j + 1] = math.cos(pos / pow(10000, j / embed_dim))
    ret.requires_grad = False
    return ret


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.n_heads
        self.attention = CausalSelfAttention(config)
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, config.expansion_factor * self.embed_dim),
            nn.GELU(),
            nn.Linear(config.expansion_factor * self.embed_dim, self.embed_dim),
            nn.Dropout(config.resid_pdrop)
        )

    def forward(self, x):
        x = x + self.attention(x)
        x = x + self.ffn(x)
        return x
    
class GPTConfig:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class GPT(nn.Module):
    '''
    GPT is a decoder-only structure
    what is needed in __init__ depends on the forward function
    '''
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim
        self.embedding = Embedding(self.vocab_size, self.embed_dim)
        # self.pe = position_embedding(self.embed_dim, self.vocab_size)
        # self.register_buffer('pe', position_embedding(self.embed_dim, self.vocab_size))
        self.pe = nn.Parameter(position_embedding(self.embed_dim, config.block_size), requires_grad=True)
        self.drop = nn.Dropout(config.resid_pdrop)
        self.ln = nn.LayerNorm(self.embed_dim)
        self.block = nn.Sequential(*[Block(config) for _ in range(config.blocks)])
        self.linear = nn.Linear(self.embed_dim, self.vocab_size)
        self.block_size = config.block_size

    def get_block_size(self):
        return self.block_size

    def forward(self, x, y=None):
        '''
        @params: x, with shape of (batch, seq_len)
        @returns: output, with shape of (batch, vocab_size). The probability distribution over the whole corpus
        '''
        batch_size, seq_len = x.shape
        embed = self.embedding(x)
        # print(f'embed.shape = {embed.shape}')
        # print(f'pe.shape = {self.pe[:, :seq_len, :].shape}')
        x = embed + self.pe[:, :seq_len, :]
        x = self.drop(x)
        x = self.block(x)
        x = self.ln(x)
        x = self.linear(x)
        print(f'x = {x.view(-1, x.size(-1))}')
        print(f'x.shape = {x.shape}')
        loss = None
        if y is not None:
            print(f'y.view(-1).shape = {y.view(-1).shape}')
            print(f'y.view(-1) = {y.view(-1)}')
            print(f'y.shape = {y.shape}')
            print('============================')
            print()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return x, loss


