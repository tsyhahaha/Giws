import os
import json
from collections import Counter

# 可调配置
data_dir = "/mnt/user/taosiyuan/projects/data/sample-submission-version/TM-training-set"  # txt文件所在目录
languages = ["chinese", "english"]
min_freq = 1  # 最小出现频率

# 特殊 tokens
special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]

def tokenize_chinese(line):
    return line.strip().split()

def tokenize_english(line):
    return line.strip().split()

def build_vocab(file_path, tokenize_fn):
    counter = Counter()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = tokenize_fn(line)
            counter.update(tokens)
    # 过滤低频词 + 排序
    vocab = [token for token, freq in counter.items() if freq >= min_freq]
    vocab = special_tokens + sorted(set(vocab))
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    return word2idx

def main():
    for lang in languages:
        file_path = os.path.join(data_dir, f"{lang}.txt")
        if lang == "chinese":
            tokenizer = tokenize_chinese
        else:
            tokenizer = tokenize_english

        print(f"Processing {file_path} ...")
        word2idx = build_vocab(file_path, tokenizer)

        save_path = os.path.join(data_dir, f"{lang}_vocab.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(word2idx, f, ensure_ascii=False, indent=2)

        print(f"Saved {lang} vocab to {save_path} (size: {len(word2idx)})")

if __name__ == "__main__":
    main()
