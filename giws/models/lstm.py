import torch
import torch.nn as nn

class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=self.num_layers)
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
        return logits, hidden