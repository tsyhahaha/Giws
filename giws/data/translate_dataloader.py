import json

import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, chinese_file, english_file, chinese_vocab_file, english_vocab_file, max_len=128):
        self.max_len = max_len
        self.alignments = None
        self.chinese_sentences = None
        self.english_sentences = None
        self.chinese_word2idx = None
        self.english_word2idx = None
        self.src_vocab_size = 0
        self.trg_vocab_size = 0

        # Load sentences
        with open(chinese_file, encoding='utf-8') as f:
            self.chinese_sentences = [line.strip().split() for line in f]

        with open(chinese_vocab_file, 'r', encoding='utf-8') as f:
            self.chinese_word2idx = json.load(f)

        with open(english_file, encoding='utf-8') as f:
            self.english_sentences = [line.strip().split() for line in f]

        with open(english_vocab_file, 'r', encoding='utf-8') as f:
            self.english_word2idx = json.load(f)

        self.src_vocab_size = len(self.chinese_word2idx)
        self.trg_vocab_size = len(self.english_word2idx)

        assert len(self.chinese_sentences) == len(self.english_sentences), \
            "Mismatch in number of sentences."
        
    def get_word2idx(self, target='src'):
        if target == 'src':
            return self.chinese_word2idx
        elif target == 'trg':
            return self.english_word2idx
            
    def __len__(self):
        return len(self.chinese_sentences)
    
    def _pad(self, indices):
        padded = indices[:self.max_len]
        padded += [self.chinese_word2idx['<pad>']] * (self.max_len - len(padded))
        return padded
    
    def _one_hot(self, sequence):
        if not isinstance(sequence, torch.LongTensor):
            sequence = sequence.long()
        
        sequence = sequence.unsqueeze(0)  # [1, seq_len]
        one_hot = torch.zeros(1, sequence.size(1), self.trg_vocab_size)
        one_hot.scatter_(2, sequence.unsqueeze(-1), 1)

        return one_hot.squeeze(0)
        
    def __getitem__(self, idx):
        chinese_sent = ['<bos>'] + self.chinese_sentences[idx] + ['<eos>']
        chinese_indices = [self.chinese_word2idx.get(word, self.chinese_word2idx['<unk>']) 
                          for word in chinese_sent]
        
        english_sent = ['<bos>'] + self.english_sentences[idx] + ['<eos>']
        english_indices = [self.english_word2idx.get(word, self.english_word2idx['<unk>']) 
                            for word in english_sent]

        return {
            'src': torch.tensor(self._pad(chinese_indices)),
            'trg': torch.tensor(self._pad(english_indices)),
        }
