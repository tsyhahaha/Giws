import sys
import pdb
import argparse
from PIL import Image

sys.path.append('/home/taosiyuan/Giws')

import torch
import torch.nn.functional as F
from peft import get_peft_config, get_peft_model, LoraConfig

from giws.CFIT.model import utils
from giws.CFIT.model.model import TwitterClassifier
import  giws.CFIT.clip as clip

device=torch.device('cuda:0')
clip_model, processer = utils.get_clip(device=device)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_file', type=str)
    parser.add_argument('--text_file', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--model_path', type=str)
    # parser.add_argument('--clip_model_path', type=str)
    # parser.add_argument('--bert_model_path', type=str)
    # parser.add_argument('--vit_model_path', type=str)
    args = parser.parse_args()
    return args

def load_model(model_path, device):
    model_info = torch.load(model_path)
    # ['model', 'cfg', 'train_steps', 'best_acc']
    model_params = model_info['model']
    cfg = model_info['cfg']
    model = TwitterClassifier(**cfg)
    if cfg.apply_lora:
        lora_config = cfg.lora_config
        peft_config = LoraConfig(
                    r=lora_config.get('lora_r', 4),
                    lora_alpha=lora_config.get('lora_alpha', 32),
                    lora_dropout=lora_config.get('lora_dropout', 0.1),
                    bias=lora_config.get('lora_bias', 'none'),
                    target_modules=['query', 'value'],
                    modules_to_save=lora_config.get('modules_to_save', 'classifier')
                )
        model = get_peft_model(model, peft_config)
    model.load_state_dict(model_params)
    return model.to(device)

def inference(model, img_file=None, text_file=None, device='cpu'):
    model.eval()
    if img_file is not None:
        img = processer(Image.open(img_file)).unsqueeze(0).to(device)
    if text_file is not None:
        with open(text_file, 'r', encoding='gb2312') as f: # utf-8, gbk, gb2312
            text = f.read()
            print("text:",text)
            text = clip.tokenize([text]).to(device)
    if name == 'clip':
        result_logits = model(img, text)
    elif name == 'bert':
        result_logits = model(text=text)
    elif name in ['vit', 'resnet']:
        result_logits = model(img=img)
    result = F.softmax(result_logits, dim=1).squeeze(0)
    # print('classify probs:', result)
    classes = ['消极', '中性', '积极']
    for idx, logits in enumerate(result):
        print(f'Class {classes[idx]}: Logits = {logits.item()}')
    return result

if __name__=='__main__':
    args = parse()
    model = load_model(args.model_path, device)
    logits = inference(model, args.img_file, args.text_file, device=device)
    
