import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from attention import MultiheadAttention


def _get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "tanh":
        return nn.Tanh
    else:
        return None


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforwad=1024,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear_1 = nn.Linear(d_model, dim_feedforwad)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(dim_feedforwad, d_model)

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def pos_embed(self, tensor, pos=None):
        return tensor if pos else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        q = k = self.pos_embed(src, pos)
        src_attn = self.self_attn(
            q,
            k,
            value=src,
            attn_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
        )[0]
        src = src + self.dropout_1(src_attn)
        src = self.norm_1(src)
        src_2 = self.linear_2(self.dropout(self.activation(self.linear_1(src))))
        src = src + self.dropout_2(src_2)
        src = self.norm_2(src)
        return src

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        src_2 = self.norm_1(src)
        q = k = self.pos_embed(src_2, pos)
        src_2 = self.self_attn(
            q, k, value=src_2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = self.dropout_1(src_2)
        src_2 = self.norm_2(src)
        src = self.linear_2(self.dropout(self.activation(self.linear_1(src_2))))
        src = src + self.dropout_2(src_2)
        return src

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear_1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(dim_feedforward, d_model)

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(d_model)
        self.dropout_2 = nn.Dropout(d_model)
        self.dropout_3 = nn.Dropout(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        q = k = self.pos_embed(tgt, query_pos)
        tgt_2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout_1(tgt_2)
        tgt = self.norm_1(tgt)
        tgt_2 = self.multihead_attn(
            query=self.pos_embed(tgt, query_pos),
            key=self.pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            tgt_key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout_2(tgt_2)
        tgt = self.norm_2(tgt)
        tgt_2 = self.linear_2(self.dropout(self.activation(self.linear_1(tgt))))
        tgt = self.norm_3(tgt)
        return tgt

        


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        normalize_before=True,
    ):

        super(Transformer, self).__init__()
        encoder_layer = TransformerEncoderLayer
