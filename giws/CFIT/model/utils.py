import os
import sys
import pdb
import torch

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

sys.path.append('../cn_clip')
from clip import utils

def get_clip(name=None, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name is None:
         
        # clip_model, preprocess = utils.load_from_name(name="/home/taosiyuan/Giws/giws/CFIT/model/pretrained/clip_cn_vit-b-16.pt", 
        #     device=device, vision_model_name='ViT-B-16', text_model_name='RoBERTa-wwm-ext-base-chinese', input_resolution=224)
        
        # clip_model, preprocess = utils.load_from_name(name="/home/taosiyuan/Giws/giws/CFIT/model/pretrained/clip_cn_rn50.pt",
        #     device=device, vision_model_name='RN50', text_model_name='RBT3-chinese', input_resolution=224)
        
        clip_model, preprocess = utils.load_from_name(name="/home/taosiyuan/Giws/giws/CFIT/model/pretrained/clip_cn_vit-h-14.pt", 
            device=device, vision_model_name='ViT-H-14', text_model_name='RoBERTa-wwm-ext-large-chinese', input_resolution=224)
    return clip_model, preprocess

def get_features(clip_model, image, text):
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return image_features, text_features

def get_train_clip(name=None, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name is None:
        
        # clip_model = utils.load_for_train(name="/home/taosiyuan/Giws/giws/CFIT/model/pretrained/clip_cn_vit-b-16.pt", 
        #     device=device, vision_model_name='ViT-B-16', text_model_name='RoBERTa-wwm-ext-base-chinese', input_resolution=224)
        
        # clip_model, preprocess = utils.load_from_name(name="/home/taosiyuan/Giws/giws/CFIT/model/pretrained/clip_cn_rn50.pt",
        #         device=device, vision_model_name='RN50', text_model_name='RBT3-chinese', input_resolution=224)
        
        clip_model, preprocess = utils.load_from_name(name="/home/taosiyuan/Giws/giws/CFIT/model/pretrained/clip_cn_vit-h-14.pt",
            device=device, vision_model_name='ViT-H-14', text_model_name='RoBERTa-wwm-ext-large-chinese', input_resolution=224)
    return clip_model.to(torch.float32)
