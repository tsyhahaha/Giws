import os
import json
from collections import Counter

data_dir = "/mnt/user/taosiyuan/projects/data/sample-submission-version/TM-training-set"
languages = ["chinese", "english"]
min_freq = 1

special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]

def tokenize(line):
    return line.strip().split()

def build_combined_vocab(file_paths, tokenize_fn):
    counter = Counter()
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = tokenize_fn(line)
                counter.update(tokens)
    vocab = [token for token, freq in counter.items() if freq >= min_freq]
    vocab = special_tokens + sorted(set(vocab))
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    return word2idx

def main():
    file_paths = [os.path.join(data_dir, f"{lang}.txt") for lang in languages]
    print(f"Processing files: {file_paths} ...")

    word2idx = build_combined_vocab(file_paths, tokenize)

    save_path = os.path.join(data_dir, "vocab.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(word2idx, f, ensure_ascii=False, indent=2)

    print(f"Saved combined vocab to {save_path} (size: {len(word2idx)})")

if __name__ == "__main__":
    main()
