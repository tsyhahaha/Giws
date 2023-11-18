import torch.nn as nn
import torch
import torch.nn.functional as F

from . import utils
from . import hook
import logging
import pdb

class CrossAttention(nn.Module):
    def __init__(self, img_input_size, text_input_size):
        super(CrossAttention, self).__init__()
        self.img_input_size = img_input_size
        self.text_input_size = text_input_size

        self.img_q = nn.Linear(img_input_size, img_input_size)
        self.img_k = nn.Linear(img_input_size, text_input_size)
        self.img_v = nn.Linear(img_input_size, img_input_size)
        self.text_q = nn.Linear(text_input_size, text_input_size)
        self.text_k = nn.Linear(text_input_size, img_input_size)
        self.text_v = nn.Linear(text_input_size, text_input_size)
        
    def forward(self, img, text):
        # pdb.set_trace()
        b, seq_len, _ = text.size()
        b, img_len, _ = img.size()

        img_query = self.img_q(img).view(b, img_len, -1)
        img_key = self.img_k(img).view(b, img_len, -1)
        img_value = self.img_v(img).view(b, img_len, -1)
        text_query = self.text_q(text).view(b, seq_len, -1)
        text_key = self.text_k(text).view(b, seq_len, -1)
        text_value = self.text_v(text).view(b, seq_len, -1)

        img_attn_scores = torch.matmul(text_query, img_key.transpose(-2, -1)) / (img_key.shape[-1] ** 0.5)
        img_attn_weights = F.softmax(img_attn_scores, dim=-1)
        img_output = torch.matmul(img_attn_weights, img_value)

        text_attn_scores = torch.matmul(img_query, text_key.transpose(-2, -1)) / (text_key.shape[-1] ** 0.5)
        text_attn_weights = F.softmax(text_attn_scores, dim=-1)
        text_output = torch.matmul(text_attn_weights, text_value)

        return img_output, text_output
        


class TwitterClassifier(nn.Module):
    def __init__(self, name='clip', img_input_size=1024, text_input_size=1024, output_size=3, use_cross_attn=False, *args, **kwargs):
        super(TwitterClassifier, self).__init__()
        self.name=name
        self.use_cross_attn = use_cross_attn
        self.clip = utils.get_train_clip()
        self.img_input_size = img_input_size
        self.text_input_size = text_input_size
        if name=='clip':
            logging.info('Both text and visual model will be used!')
            self.input_size = img_input_size + text_input_size
        elif name=='bert':
            logging.info('Only text side model will be used!')
            self.input_size = text_input_size
        elif name=='vit' or name=='resnet':
            logging.info('Only visual side model will be used!')
            self.input_size = img_input_size
        self.output_size = output_size
        
        if self.use_cross_attn:
            self._cross_attn = CrossAttention(img_input_size // 32, text_input_size // 32) 
        self.classifier = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
        self.gradient_checker = hook.GradientChecker(self)

    def forward(self, img, text):
        if self.name == 'clip':
            img_feature, text_feature = self.clip(img, text)
            b = img_feature.shape[0]
            # single value cross attn
            """
            img_weight = F.softmax(torch.matmul(img_feature, text_feature.t()))
            text_weight = F.softmax(torch.matmul(text_feature, img_feature.t()))
            input_ids = torch.cat((
                torch.matmul(img_weight, img_feature),
                torch.matmul(text_weight, text_feature))
                , dim=1)
            """
            if self.use_cross_attn:
                _img_feature = img_feature.view(b, 32, -1)
                _text_feature = text_feature.view(b, 32, -1)
                img_output, text_output = self._cross_attn(_img_feature, _text_feature)
                input_ids = torch.cat((img_output.view(b, -1), text_output.view(b, -1)), dim=1)
            else:
                input_ids = torch.cat((img_feature, text_feature), dim=1)
        elif self.name == 'bert':
            input_ids = self.clip.encode_text(text)
        elif self.name=='vit' or self.name == 'resnet':
            input_ids = self.clip.encode_image(img)
        return self.classifier(input_ids)

