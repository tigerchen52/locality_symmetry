# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# (Strongly inspired by original Google BERT code and Hugging Face's code)

""" Transformer Model Classes & Config Class """

import math
import json
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import split_last, merge_last


class Config(NamedTuple):
    "Configuration for BERT model"
    vocab_size: int = None # Size of Vocabulary
    dim: int = 768 # Dimension of Hidden Layer in Transformer Encoder
    n_layers: int = 12 # Numher of Hidden Layers
    n_heads: int = 12 # Numher of Heads in Multi-Headed Attention Layers
    dim_ff: int = 768*4 # Dimension of Intermediate Layers in Positionwise Feedforward Net
    #activ_fn: str = "gelu" # Non-linear Activation Function Type in Hidden Layers
    p_drop_hidden: float = 0.1 # Probability of Dropout of various Hidden Layers
    p_drop_attn: float = 0.1 # Probability of Dropout of Attention Layers
    max_len: int = 512 # Maximum Length for Positional Embeddings
    n_segments: int = 2 # Number of Sentence Segments

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.dim))
        self.beta  = nn.Parameter(torch.zeros(cfg.dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings."
    def __init__(self, cfg):
        super().__init__()
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim) # token embedding
        self.pos_embed = nn.Embedding(cfg.max_len, cfg.dim) # position embedding
        self.seg_embed = nn.Embedding(cfg.n_segments, cfg.dim) # segment(token type) embedding

        self.norm = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, seg):
        #seq_len = x.size(1)
        #pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        #pos = pos.unsqueeze(0).expand_as(x) # (S,) -> (B, S)

        e = self.tok_embed(x) + self.seg_embed(seg)
        return self.drop(self.norm(e))


class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.dim, cfg.dim)
        self.proj_k = nn.Linear(cfg.dim, cfg.dim)
        self.proj_v = nn.Linear(cfg.dim, cfg.dim)
        self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None # for visualization
        self.n_heads = cfg.n_heads

    def forward(self, x, mask, pos):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        scores += pos[:, None, :, :]

        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PosAttention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(math.sqrt(1 / math.sqrt(cfg.dim))))
        self.projection = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim),
        )
        self.dropout = nn.Dropout(p=cfg.p_drop_hidden)

    def _attention(self, a, b):
        return torch.matmul(a, b.transpose(1, 2)) * self.temperature

    def forward(self, x, mask, pos_att):
        mask = mask.unsqueeze(-1).bool().repeat(1, 1, list(pos_att.size())[-1])
        pos_att.masked_fill_(~mask, 0.)
        pos_feature = torch.matmul(pos_att.transpose(1, 2), x)
        pos_feature = self.projection(pos_feature)
        return self.dropout(pos_feature)


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.dim, cfg.dim_ff)
        self.fc2 = nn.Linear(cfg.dim_ff, cfg.dim)
        #self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))


class PositionalAttCached(nn.Module):
    def __init__(self, d_model, pos_attns):
        super(PositionalAttCached, self).__init__()
        # Compute the positional encodings once in log space.
        self.d_model = d_model
        self.pos_attns = pos_attns
        #self.max_len = max_len

    def forward(self, x, p_type=None):
        shape = list(x.size())
        pos_attn = self.pos_attns[shape[1]]
        p_e = Variable(pos_attn, requires_grad=False).cuda()
        p_e = p_e.repeat([shape[0], 1, 1])
        return p_e


def get_pre_cal_pa(strategy, max_len, dim=200, window_size=3, w=0.5):
    if strategy == 'fixed':
        return cal_fixed_pos_att(max_len, window_size)
    if strategy == 'gaussian':
        return cal_guassian_att(max_len, w=w)
    return cal_pos_att(max_len, dim)


def get_pos_endcode(d_model, max_len=5000):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    position = position * 1
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


def cal_pos_att(max_len, dim):
    attn_dict = dict()
    for sen_len in range(1, max_len+1):
        position = get_pos_endcode(dim, sen_len)
        a, b = torch.FloatTensor(position), torch.FloatTensor(position)
        temperature = nn.Parameter(torch.tensor(math.sqrt(1 / math.sqrt(dim))))
        attn = torch.matmul(a, b.transpose(0, 1)) * temperature
        attn = F.softmax(attn, dim=1)
        attn_dict[sen_len] = attn
    return attn_dict


def cal_fixed_pos_att(max_len, window_size):
    win = (window_size - 1) // 2
    weight = float(1 / window_size)
    attn_dict = dict()
    for sen_len in range(1, max_len+1):
        attn = np.eye(sen_len)
        if sen_len < window_size:
            attn_dict[sen_len] = attn
            continue
        for i in range(sen_len):
            attn[i, i-win:i+win+1] = weight
        attn[0, 0:win+1] = weight
        attn_dict[sen_len] = torch.FloatTensor(attn)

    return attn_dict


def cal_guassian_att(max_len, w=0.5):

    attn_dict = dict()
    for sen_len in range(1, max_len+1):
        if sen_len > 1:
            attn = np.zeros((sen_len, sen_len))
            for i in range(sen_len):
                for j in range(sen_len):
                    attn[i][j] = 1.0 * (j - i)
                attn[i] = [- abs(w * v ** 2) for v in attn[i]]
                attn[i] = np.exp(attn[i]) / sum(np.exp(attn[i]))
        else:
            attn = np.array([1.0])
        attn_dict[sen_len] = torch.FloatTensor(attn)
    return attn_dict


pos_attns = get_pre_cal_pa('gaussian', max_len=512, dim=192)


class Block(nn.Module):
    """ Transformer Block """
    def __init__(self, cfg, p_type):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.pos_attns = pos_attns
        self.p_type = p_type

    def forward(self, x, mask):
        shape = list(x.size())
        position = PositionalAttCached(shape[-1], self.pos_attns)
        pos_att = position(x, self.p_type)

        h = self.attn(x, mask, pos_att)
        h = self.norm1(x + self.drop(self.proj(h)))
        h = self.norm2(h + self.drop(self.pwff(h)))
        return h


class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""
    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings(cfg)
        self.blocks = nn.ModuleList([Block(cfg, p_type=None) for _ in range(cfg.n_layers)])

    def forward(self, x, seg, mask):
        h = self.embed(x, seg)
        for block in self.blocks:
            h = block(h, mask)
        return h

