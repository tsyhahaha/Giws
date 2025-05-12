import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Union
from einops import rearrange

class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 embed_dim,
                 n_blocks,
                 n_heads,
                 ff_hid_dim,
                 max_length,
                 dropout,
                 device,):
        super().__init__()
        self.encoder = Encoder(src_vocab_size,
                               embed_dim,
                               n_blocks,
                               n_heads,
                               ff_hid_dim,
                               max_length,
                               dropout,
                               device,)
        self.decoder = Decoder(trg_vocab_size,
                               embed_dim,
                               n_blocks,
                               n_heads,
                               ff_hid_dim,
                               max_length,
                               dropout,
                               device,)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).bool().to(self.device) & trg_pad_mask
        return trg_mask.to(self.device)

    def forward(self, src, trg, use_efficient_attn=False):
        src_mask = self.src_mask(src)
        trg_mask = self.trg_mask(trg)
        encoded = self.encoder(src, src_mask, use_efficient_attn=use_efficient_attn)
        decoded = self.decoder(trg, encoded, trg_mask, src_mask, use_efficient_attn=use_efficient_attn)
        return decoded

# ref: https://github.com/bytedance/Protenix/blob/main/protenix/model/modules/primitives.py
def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    use_efficient_implementation: bool = False,
    attn_weight_dropout_p: float = 0.0,
    inplace_safe: bool = False,
) -> torch.Tensor:
    """Attention.

    Args:
        q (torch.Tensor): query tensor of shape [..., n_q, d]
        k (torch.Tensor): key tensor of shape [..., n_kv, d]
        v (torch.Tensor): value tensor of shape[..., n_kv, d]
        attn_bias (torch.Tensor, optional): attention bias tensor of shape [..., n_q, n_kv]. Defaults to None.
        use_efficient_implementation (bool): whether to use the torch.nn.functional.scaled_dot_product_attention, Defaults to False.
        attn_weight_dropout_p (float): Dropout probability; if greater than 0.0, dropout is applied, Defaults to 0.0.

    Returns:
        torch.Tensor: output of tensor [..., n_q, d]
    """
    assert k.shape == v.shape
    if use_efficient_implementation:
        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_bias,
            dropout_p=attn_weight_dropout_p,
        )
        return attn_output
    # [..., n_kv, d] -> [..., d, n_kv]
    k = k.transpose(-1, -2)

    # [..., n_q, d], [..., d, n_kv] -> [..., n_q, n_kv]
    attn_weights = q @ k

    if attn_bias is not None:
        if inplace_safe:
            attn_weights += attn_bias
        else:
            attn_weights = attn_weights + attn_bias

    # [..., n_q, n_kv]
    attn_weights = F.softmax(attn_weights, dim=-1)

    # [..., n_q, n_kv], [..., n_kv, d] -> [..., n_q, d]
    attn_output = attn_weights @ v

    return attn_output

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()
        self.head_dim = embed_dim // n_heads
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.scale = embed_dim ** 0.5
        self.dropout_ratio = dropout

        self.keys = nn.Linear(embed_dim, embed_dim)
        self.queries = nn.Linear(embed_dim, embed_dim)
        self.values = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, 
                use_efficient_implementation=False):
        N = q.size(0)          # batch_size
        Q = self.queries(q)    # shape: [N, query_len, embed_dim]
        K = self.keys(k)       # shape: [N, key_len, embed_dim]
        V = self.values(v)     # shape: [N, value_len, embed_dim]

        Q = Q.view(N, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # shape: [N, n_heads, query_len, head_dim]
        K = K.view(N, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # shape: [N, n_heads, key_len, head_dim]
        V = V.view(N, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # shape: [N, n_heads, value_len, head_dim]

        # Naive implementation
        # energy = (Q @ K.permute(0, 1, 3, 2)) / self.scale # [N, n_heads, key_len, query_len]
        # if mask is not None:
        #     energy = energy.masked_fill(mask == 0, torch.finfo(Q.dtype).min)

        # attention = energy.softmax(-1)           # shape: [N, n_heads, query_len, key_len]
        # x = self.dropout(attention) @ V          # shape: [N, n_heads, query_len, key_len]
        # x = x.permute(0, 2, 1, 3).contiguous()   # shape: [N, query_len, n_heads, head_dim]
        # x = x.view(N, -1, self.embed_dim)        # shape: [N, query_len, embed_dim]
        # x = self.proj(x)
        # return x

        # broadcast to [B, N_q, N_kv]
        attn_bias = None
        if mask is not None:
            assert mask.dim() == 4, f'mask\'s dim is {mask.dim()} != 4'
            attn_bias = mask.to(dtype=q.dtype)
            attn_bias = (1.0 - attn_bias) * -1e9      # padding -> -1e9， real token -> 0

        out = _attention(
                q=Q,
                k=K,
                v=V,
                attn_bias=attn_bias,
                use_efficient_implementation=use_efficient_implementation,
                attn_weight_dropout_p=self.dropout_ratio,
            )
        # Revert back to orignal shape
        out = rearrange(out, "b h s c -> b s (h c)")
        return out



class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, ff_hid_dim, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hid_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, src, mask, use_efficient_attn=False):
        attention = self.attention(src, src, src, mask, use_efficient_attn)
        x = self.norm1(attention + self.dropout(src))
        out = self.mlp(x)
        out = self.norm2(out + self.dropout(x))
        return out


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_blocks, n_heads, ff_hid_dim, max_length, dropout, device):
        super().__init__()
        self.device = device
        self.scale = embed_dim ** 0.5
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_length, embed_dim)
        self.blocks = nn.ModuleList([EncoderLayer(embed_dim, n_heads, ff_hid_dim, dropout)] * n_blocks)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask, use_efficient_attn=False):
        N, seq_len = src.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        pos_embeddings = self.pos_emb(positions)
        tok_embeddings = self.tok_emb(src.long()) * self.scale
        out = self.dropout(pos_embeddings + tok_embeddings)

        for block in self.blocks:
            out = block(out, mask, use_efficient_attn)

        return out


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, ff_hid_dim, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(embed_dim, n_heads, dropout)   # decoder self-attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.joint_attention = MultiHeadAttention(embed_dim, n_heads, dropout)  # encoder-decoder attention
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hid_dim, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask, src_mask, use_efficient_attn=False):
        trg_attention = self.self_attention(trg, trg, trg, trg_mask, use_efficient_attn)
        trg = self.norm1(trg + self.dropout(trg_attention))
        joint_attention = self.joint_attention(trg, src, src, src_mask, use_efficient_attn)
        trg = self.norm2(trg + self.dropout(joint_attention))
        out = self.mlp(trg)
        out = self.norm3(trg + self.dropout(out))
        return out


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_blocks, n_heads, ff_hid_dim, max_length, dropout, device,):
        super().__init__()
        self.device = device
        self.scale = embed_dim ** 0.5
        self.tok_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_length, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([DecoderLayer(embed_dim, n_heads, ff_hid_dim, dropout)] * n_blocks)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, trg, src, trg_mask, src_mask, use_efficient_attn=False):
        N, trg_len = trg.shape
        positions = torch.arange(0, trg_len).expand(N, trg_len).to(self.device)
        pos_embeddings = self.pos_embedding(positions)
        tok_embeddings = self.tok_embedding(trg.long()) * self.scale
        trg = self.dropout(pos_embeddings + tok_embeddings)

        for block in self.blocks:
            trg = block(trg, src, trg_mask, src_mask, use_efficient_attn)

        output = self.fc(trg)
        return output


