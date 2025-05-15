# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Translation/Transformer/README.md

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
import math

from giws.models.ops import _attention

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
    
    def generate(self, src, max_len=50, pad_idx=0, bos_idx=2, eos_idx=3, use_efficient_attn=False, beam_size=4):
        device = src.device
        batch_size = src.size(0)
        sequences = [[] for _ in range(batch_size)]

        for b in range(batch_size):
            beams = [(torch.tensor([bos_idx], device=device), 0.0)]  # [(sequence_tensor, score)]
            finished = []

            for _ in range(max_len - 1):
                all_candidates = []
                for seq, score in beams:
                    if seq[-1].item() == eos_idx:
                        finished.append((seq, score))
                        continue

                    input_seq = seq.unsqueeze(0)  # shape: [1, t]
                    out = self(src[b:b+1], input_seq, use_efficient_attn=use_efficient_attn)  # [1, t, vocab]
                    log_probs = F.log_softmax(out[0, -1], dim=-1)  # [vocab]
                    topk_probs, topk_idx = log_probs.topk(beam_size)

                    for i in range(beam_size):
                        next_seq = torch.cat([seq, topk_idx[i].unsqueeze(0)])
                        new_score = score + topk_probs[i].item()
                        all_candidates.append((next_seq, new_score))

                # 选 top beam_size 条路径
                beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]

                if all(seq[-1].item() == eos_idx for seq, _ in beams):
                    finished.extend(beams)
                    break

            # 最终选择得分最高的完成序列
            candidates = finished if finished else beams
            best_seq = sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]
            sequences[b] = best_seq

        # pad 到相同长度
        padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=pad_idx)
        return padded




class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()
        self.head_dim = embed_dim // n_heads
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.scale = self.head_dim ** 0.5
        self.dropout_ratio = dropout

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias = False)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, 
                use_efficient_implementation=False):
        N = x.size(0)          # batch_size
        qkv = self.qkv_proj(x).chunk(3, dim = -1)    # shape: [N, query_len, 3 * embed_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # broadcast to [B, N_q, N_kv]
        attn_bias = None
        if mask is not None:
            assert mask.dim() == 4, f'mask\'s dim is {mask.dim()} != 4'
            attn_bias = mask.to(dtype=q.dtype)
            attn_bias = (1.0 - attn_bias) * -1e9      # padding -> -1e9， real token -> 0

        out = _attention(
                q=q,
                k=k,
                v=v,
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
        attention = self.attention(src, mask, use_efficient_attn)
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
        self.pos_emb = PositionalEncoding(emb_dim=embed_dim, max_len=max_length)
        self.blocks = nn.ModuleList([EncoderLayer(embed_dim, n_heads, ff_hid_dim, dropout)] * n_blocks)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask, use_efficient_attn=False):
        N, seq_len = src.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        pos_embeddings = self.pos_emb(positions)
        tok_embeddings = self.tok_emb(src.long()) * self.scale
        try:
            out = self.dropout(pos_embeddings + tok_embeddings)
        except:
            print(tok_embeddings.shape, pos_embeddings.shape)
            import pdb; pdb.set_trace()

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
        self.pos_embedding = PositionalEncoding(emb_dim=embed_dim, max_len=max_length)
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
    
# ref: https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py
class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # shape (max_len, 1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))  # shape (d_model/2)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1), :]    # [max_len, (1 broadcast to bz), d_model]
    




