from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from giws.CFIT.model.model import TwitterClassifier
import pdb
import torch

model = TwitterClassifier(input_size=1024)
device = torch.device('cuda:0')
peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        bias='none',
        target_modules=['query', 'value', ]
    )
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.to(device)
