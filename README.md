# GIW
GIW（Give It a Whirl）——About Training Details

This reposiroty contains several mini projects which require diffrent training details. it‘s builded to record some training details and techniques, and most of them are of great benefit. Each point mentioned in the Readme is implemented in the projects.

The operation of the entire project is managed through **hydra**, please refer [here](https://hydra.cc/docs/intro/) for details.

## CFIT(classifier of img and text)

> Three classification tasks: according to the image and its corresponding text. Training data is from Twitter.

**Baseline**：CLIP（Bert + Res50 or ViT）

* Model & Training & Evaluation：Complete model definition, training and evaluation script writing. Comprehensive data pipeline constructing.
* Accelerate：AMP（auto mixed precision）、Accumulate Gradient.
* PEFT (parameter-efficient fine-tuning)：LoRA（use peft by huggingface）.
