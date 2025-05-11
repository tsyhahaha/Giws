from collections import Counter

import torch
from torch.utils.data import Dataset



class TranslationDataset(Dataset):
    def __init__(self, chinese_file, english_file=None, alignment_file=None, max_len=128, vocab_size=30000):
        self.max_len = max_len
        self.alignments = None
        self.chinese_sentences = None
        self.english_sentences = None

        # Load sentences
        with open(chinese_file, encoding='utf-8') as f:
            self.chinese_sentences = [line.strip().split() for line in f]

        with open(english_file, encoding='utf-8') as f:
            self.english_sentences = [line.strip().split() for line in f]
        if alignment_file is not None:
            with open(alignment_file, encoding='utf-8') as f:
                self.alignments = [line.strip() for line in f]

        assert len(self.chinese_sentences) == len(self.english_sentences), \
            "Mismatch in number of sentences."
            
        # Cn-vocab
        chinese_words = [word for sent in self.chinese_sentences for word in sent]
        self.chinese_vocab = ['<pad>', '<sos>', '<eos>', '<unk>'] + \
               [word for word, _ in Counter(chinese_words).most_common(vocab_size)]  # 限制词
        
        # En-vocab
        english_words = [word for sent in self.english_sentences for word in sent]
        self.english_vocab = ['<pad>', '<sos>', '<eos>', '<unk>'] + \
               [word for word, _ in Counter(english_words).most_common(vocab_size)]
        
        # mapping from words to idx
        self.chinese_word2idx = {word: idx for idx, word in enumerate(self.chinese_vocab)}
        self.english_word2idx = {word: idx for idx, word in enumerate(self.english_vocab)}

    def __len__(self):
        return len(self.chinese_sentences)
    
    def _pad(self, indices):
        padded = indices[:self.max_len]
        padded += [self.chinese_word2idx['<pad>']] * (self.max_len - len(padded))
        return padded
    
    def _one_hot(self, sequence, vocab_size=30000):
        if not isinstance(sequence, torch.LongTensor):
            sequence = sequence.long()
        
        sequence = sequence.unsqueeze(0)  # 变为 [1, seq_len]
        one_hot = torch.zeros(1, sequence.size(1), vocab_size)
        one_hot.scatter_(2, sequence.unsqueeze(-1), 1)
        return one_hot.squeeze(0)  # 移除 batch 维度 -> [seq_len, vocab_size]
        
    def __getitem__(self, idx):
        chinese_sent = ['<sos>'] + self.chinese_sentences[idx] + ['<eos>']
        chinese_indices = [self.chinese_word2idx.get(word, self.chinese_word2idx['<unk>']) 
                          for word in chinese_sent]
        
        english_sent = ['<sos>'] + self.english_sentences[idx] + ['<eos>']
        english_indices = [self.english_word2idx.get(word, self.english_word2idx['<unk>']) 
                            for word in english_sent]

        return {
            'src': torch.tensor(self._pad(chinese_indices)),
            'trg': torch.tensor(self._pad(english_indices)),
            'trg_one_hot': self._one_hot(torch.tensor(self._pad(english_indices))),
        }
