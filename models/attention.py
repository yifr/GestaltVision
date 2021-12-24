import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / np.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, *embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class SlotAttention(nn.Module):
    def __init__(
        self, num_iterations, num_slots, slot_dim, mlp_hidden_dim, epsilon=1e-8
    ):
        """
        Slot attention
        Params:
            num_iterations: int: how many iterations to pay attention over
            num_slots: int: how many slots to initialize
            slot_dim: int: dimensions of each slot
            mlp_hidden_dim: int: size of hidden layer in mlp
        """
        super(SlotAttention, self).__init__()

        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.epsilon = epsilon

        self.queries = nn.Linear(slot_dim, slot_dim)
        self.keys = nn.Linear(slot_dim, slot_dim)
        self.values = nn.Linear(slot_dim, slot_dim)

        self.input_norm = nn.LayerNorm(slot_dim)
        self.mlp_norm = nn.LayerNorm(slot_dim)
        self.slot_norm = nn.LayerNorm(slot_dim)

        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, slot_dim),
        )

        self.gru = nn.GRUCell(slot_dim, slot_dim)

        self.slots_mu = torch.empty((1, 1, slot_dim))
        self.slots_log_sigma = torch.empty((1, 1, slot_dim))
        nn.init.xavier_uniform_(self.slots_mu)
        nn.init.xavier_uniform_(self.slots_log_sigma)

    def forward(self, x, slot_initialization=None):
        b, d = x.shape[0], x.shape[-1]

        x = self.input_norm(x)
        keys = self.keys(x)  # [batch_size, input_size, slot_dim]
        values = self.values(x)  # [batch_size, input_size, slot_dim]

        mu = self.slots_mu.expand(b, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(b, self.num_slots, -1)

        if slot_initialization is not None:
            slots = slot_initialization
        else:
            slots = mu + sigma * torch.randn(
                mu.shape, device=x.device
            )  # [batch_size, num_slots, slot_dim]

        for i in range(self.num_iterations):
            slots_prev = slots
            slots = self.slot_norm(slots)  # [batch_size, num_slots, slot_dim]

            # Compute attention
            queries = self.queries(slots)  # [batch_size, num_slots, slot_dim]

            # b = batch_size, i = input_size, d = slot_dim, k = num_slots
            attn_logits = torch.einsum(
                "bid,bkd->bik", queries, keys
            )  # [batch_size, input_size, num_slots]
            attn_logits *= self.slot_dim ** -0.5
            attn = F.softmax(attn_logits, dim=1)

            # Weighted Mean
            attn += self.epsilon
            attn /= torch.sum(attn, dim=-1, keepdim=True)

            updates = torch.einsum(
                "bkd,bik->bid", values, attn
            )  # [batch_size, num_slots, slot_dim]

            updates = updates.reshape(-1, d)
            slots_prev = slots_prev.reshape(-1, d)

            slots = self.gru(updates, slots_prev)

            slots = slots.reshape(b, -1, d)
            slots += self.mlp(self.mlp_norm(slots))

        return slots
