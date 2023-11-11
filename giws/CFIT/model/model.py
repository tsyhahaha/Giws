import torch.nn as nn
import torch

class TwitterClassifier(nn.Module):
    def __init__(self, clip_model, input_size=2048, output_size=3):
        super(TwitterClassifier, self).__init__()
        self.clip = clip_model
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, img, text):
        img_feature, text_feature, sim = self.clip(img, text)
        input_ids = torch.cat((img_feature, text_feature, sim), dim=1).to(torch.float32)
        return self.fc(input_ids)



