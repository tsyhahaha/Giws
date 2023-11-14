import torch.nn as nn
import torch

from . import utils
from . import hook
import pdb

class TwitterClassifier(nn.Module):
    def __init__(self, input_size=2048, output_size=3):
        super(TwitterClassifier, self).__init__()
        self.clip = utils.get_train_clip()

        self.classifier = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        self.gradient_checker = hook.GradientChecker(self)

    def forward(self, img, text):
        img_feature, text_feature = self.clip(img, text)
        
        # sim_score = self._get_sim(img, text)
        # input_ids = torch.cat((img_feature, text_feature, sim_score.unsqueeze(1)), dim=1).to(torch.float32)
        
        input_ids = torch.cat((img_feature, text_feature),dim=1).to(torch.float32)
        return self.classifier(input_ids)

    def _get_sim(self, img, text):
        with torch.no_grad():
            img_sim, text_sim = self.clip.get_similarity(img, text)
        text_score = text_sim.diag()
        return text_score


