import giws.CFIT.model.utils as utils
import giws.CFIT.clip as clip

import torch
from PIL import Image
import pdb

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = utils.get_clip(device=device)
model.eval()
# 0001.jpg  0113.jpg  0225.jpg
img1 = preprocess(Image.open("/home/taosiyuan/DataSet/train/img/0001.jpg")).unsqueeze(0)
img2 = preprocess(Image.open("/home/taosiyuan/DataSet/train/img/0113.jpg")).unsqueeze(0)
img3 = preprocess(Image.open("/home/taosiyuan/DataSet/train/img/0225.jpg")).unsqueeze(0)
image = torch.cat([img1, img2, img3], dim=0).to(device)

with open('/home/taosiyuan/DataSet/train/text/0001.txt', 'r') as f1:
    text1 = f1.read()
with open('/home/taosiyuan/DataSet/train/text/0113.txt', 'r') as f2:
    text2 = f2.read()
with open('/home/taosiyuan/DataSet/train/text/0225.txt', 'r') as f3:
    text3 = f3.read()
text = clip.tokenize([text1, text2, text3]).to(device)

with torch.no_grad():
    img_feature, text_feature, probs = model(image, text)
    pdb.set_trace()

    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
    image_features /= image_features.norm(dim=-1, keepdim=True) 
    text_features /= text_features.norm(dim=-1, keepdim=True)    

    logits_per_image, logits_per_text = model.get_similarity(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
print("Label probs:", probs)

