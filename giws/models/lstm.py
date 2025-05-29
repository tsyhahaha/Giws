import torch
import torch.nn as nn

import math
from einops import rearrange

class NaiveCustomLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, batch_first=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.weights = nn.ModuleList()

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size

            cell = nn.ParameterDict({
                'W_f': nn.Parameter(torch.Tensor(layer_input_size, hidden_size)),
                'U_f': nn.Parameter(torch.Tensor(hidden_size, hidden_size)),
                'b_f': nn.Parameter(torch.Tensor(hidden_size)),

                'W_i': nn.Parameter(torch.Tensor(layer_input_size, hidden_size)),
                'U_i': nn.Parameter(torch.Tensor(hidden_size, hidden_size)),
                'b_i': nn.Parameter(torch.Tensor(hidden_size)),

                'W_c': nn.Parameter(torch.Tensor(layer_input_size, hidden_size)),
                'U_c': nn.Parameter(torch.Tensor(hidden_size, hidden_size)),
                'b_c': nn.Parameter(torch.Tensor(hidden_size)),

                'W_o': nn.Parameter(torch.Tensor(layer_input_size, hidden_size)),
                'U_o': nn.Parameter(torch.Tensor(hidden_size, hidden_size)),
                'b_o': nn.Parameter(torch.Tensor(hidden_size)),
            })

            self.weights.append(cell)

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for layer in self.weights:
            for name, param in layer.items():
                param.data.uniform_(-stdv, stdv)

    def forward(self, x, init_states=None):
        """
        Forward pass for multi-layer LSTM.

        Instead of computing W @ [h_t, x_t] + b via concatenation,
        we decompose it into separate projections:
            [W, U]^T @ [h_t, x_t] = W @ h_t + U @ x_t
        """
        if self.batch_first:
            bs, seq_sz, _ = x.size()
        else:
            seq_sz, bs, _ = x.size()
        hidden_seq = x

        if init_states is None:
            h_t = [torch.zeros(bs, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
            c_t = [torch.zeros(bs, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        else:
            h_t, c_t = init_states

        new_h, new_c = [], []

        for layer in range(self.num_layers):
            layer_out = []
            h, c = h_t[layer], c_t[layer]
            params = self.weights[layer]

            for t in range(seq_sz):
                if self.batch_first:
                    x_t = hidden_seq[:, t, :]
                else:
                    x_t = hidden_seq[t]

                f_t = torch.sigmoid(x_t @ params['W_f'] + h @ params['U_f'] + params['b_f'])
                i_t = torch.sigmoid(x_t @ params['W_i'] + h @ params['U_i'] + params['b_i'])
                g_t = torch.tanh(x_t @ params['W_c'] + h @ params['U_c'] + params['b_c'])
                c = f_t * c + i_t * g_t
                o_t = torch.sigmoid(x_t @ params['W_o'] + h @ params['U_o'] + params['b_o'])
                h = o_t * torch.tanh(c)

                layer_out.append(h.unsqueeze(0))
            if self.batch_first:
                hidden_seq = torch.cat(layer_out, dim=0).transpose(0, 1).contiguous()
            else:
                hidden_seq = torch.cat(layer_out, dim=0)
            new_h.append(h)
            new_c.append(c)

        return hidden_seq, (new_h, new_c)
    

class PTBModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers,):
        super(PTBModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = NaiveCustomLSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers)
        self.proj_o = nn.Linear(in_features=hidden_dim, out_features=vocab_size)

    def forward(self, seq):
        seq_emb = self.emb(seq)
        hidden_seq, (new_h, new_c) = self.lstm(seq_emb)
        logits = self.proj_o(hidden_seq)  # [N, S, D] -> [N, S, vocab_size]
        return logits, (new_h, new_c)


class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=self.num_layers)
        self.lstm = NaiveCustomLSTM(embedding_dim, self.hidden_dim, num_layers=self.num_layers)
        self.linear = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input_ids, hidden=None):
        seq_len, batch_size = input_ids.size()

        # initiailize hidden state
        if hidden is None:
            h_0 = input_ids.data.new(self.num_layers, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input_ids.data.new(self.num_layers, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden

        embeds = self.embeddings(input_ids)
        output, hidden = self.lstm(embeds, (h_0, c_0))
        logits = self.linear(output.view(seq_len * batch_size, -1))
        logits = rearrange(logits, '(s b) v -> s b v', b=batch_size)
        return logits, hidden
    

