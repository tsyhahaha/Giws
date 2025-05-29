import torch
from torch.utils.data import Dataset

import os
import json


def read_words(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read().replace("\n", "<eos>").split()

def build_vocab(filename):
    words = read_words(filename)
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    sorted_words = sorted(word_freq.items(), key=lambda x: (-x[1], x[0]))
    word_to_id = {word: idx for idx, (word, _) in enumerate(sorted_words)}
    return word_to_id

def file_to_word_ids(filename, word_to_id):
    words = read_words(filename)
    return [word_to_id.get(word, word_to_id["<unk>"]) for word in words]

def load_ptb_data(data_path):
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    
    word_to_id = build_vocab(train_path)
    if "<unk>" not in word_to_id:
        raise ValueError("Vocab must include '<unk>' token.")

    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)

    id_to_word = {v: k for k, v in word_to_id.items()}
    
    return train_data, valid_data, word_to_id, id_to_word


class PTBDataset(Dataset):
    def __init__(self, raw_data, batch_size, seq_len):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len

        data = torch.tensor(raw_data, dtype=torch.long)

        data_len = len(data)
        batch_len = data_len // batch_size

        # drop_last
        data = data[:batch_size * batch_len]
        self.data = data.view(batch_size, batch_len)

        self.data_num = (batch_len - 1) // seq_len  

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        """
        - x: [batch_size, seq_len]
        - y: [batch_size, seq_len], the target sequence of x, shifted right by 1 position.
        """
        i = idx * self.seq_len
        x = self.data[:, i     : i + self.seq_len]
        y = self.data[:, i + 1 : i + self.seq_len + 1]
        return x, y


if __name__=='__main__':
    train_data, valid_data, word_to_id, id_to_word = load_ptb_data("/mnt/user/taosiyuan/projects/data/simple-examples/data")

    print("Vocabulary size:", len(word_to_id))
    print("Train sample:", train_data[:20])

    # save the vocab dict for inference
    with open("/mnt/user/taosiyuan/projects/data/simple-examples/data/vocab.json", "w", encoding="utf-8") as f:
        json.dump(word_to_id, f, ensure_ascii=False, indent=4)