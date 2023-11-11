import os
import sys
import pdb
import torch

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

sys.path.append('../cn_clip')
from clip import utils

def get_clip(device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_model, preprocess = utils.load_from_name(name="/root/autodl-tmp/clip_cn_rn50.pt", device=device, vision_model_name='RN50', text_model_name='RBT3-chinese', input_resolution=224)
    return clip_model, preprocess

def get_features(clip_model, image, text):
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return image_features, text_features

