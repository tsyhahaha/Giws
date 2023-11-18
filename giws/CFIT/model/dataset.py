import os
import sys
from PIL import Image

import clip as clip

import torch
from torch.utils.data import Dataset

class GTDataset(Dataset):
    def __init__(self, img_folder, text_folder, label_folder, image_processer=None):
        self.data = []
        self.class_num = [0,0,0]
        assert os.path.isdir(img_folder) and os.path.isdir(text_folder) and os.path.isdir(label_folder), f'check your data folder path: {img_folder}, {text_folder}, {label_folder}'
        img_files = sorted(os.listdir(img_folder))
        text_files = sorted(os.listdir(text_folder))
        label_files = sorted(os.listdir(label_folder))
        for i in range(len(img_files)):
            if image_processer is not None:
                _img = os.path.join(img_folder, img_files[i])
                try:
                    image = image_processer(Image.open(_img)).squeeze(0)
                except Exception as e:
                    print(f"Loading {_img} error: {e}")
                    continue
            else:
                print('you need a image processer!')
                image = None
            with open(os.path.join(text_folder, text_files[i]), 'r', errors='ignore') as f:
                content = f.read()
                text = clip.tokenize(content).squeeze(0)
            with open(os.path.join(label_folder, label_files[i]), 'r', errors='ignore') as f:
                label = torch.zeros(3,)
                label_ids = int(f.read().split()[0]) + 1
                label[label_ids] = 1
                self.class_num[label_ids] += 1

            self.data.append((image, text, label))


    def get_class_num(self):
        class_num = [0,0,0]
        for i, j, label in self.data:
            indics = torch.argmax(label)
            class_num[indics] += 1
        return class_num

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

