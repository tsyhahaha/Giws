# GIW
GIW（Give It a Whirl）——About Training Details

This reposiroty contains several mini projects which require diffrent training details. it‘s builded to record some training details and techniques, and most of them are of great benefit. Each point mentioned in the Readme is implemented in the projects.

The operation of the entire project is managed through **hydra**, please refer [here](https://hydra.cc/docs/intro/) for details.

## CFIT(classifier of img and text)

> Sentiment Classification tasks(three types of motion): according to the image and its corresponding text. Training data is from Twitter.

**Baseline**：CLIP（Bert + Res50 or ViT）

* Model & Training & Evaluation：Complete model definition, training and evaluation script writing. Comprehensive data pipeline constructing.
* Accelerate：AMP（auto mixed precision）、Accumulate Gradient、DDP（Distributed Data Parallel）
* PEFT (parameter-efficient fine-tuning)：LoRA（use peft by huggingface）.

**Memory**

AMP and AG is applied by defalt.

| Model | Config           | Memory usage |
| ----- | ---------------- | ------------ |
| CLIP  | Full Fine-tuning | 27.6 G       |
| CLIP  | LoRA             | 9.3 G        |
| CLIP  | LoRA + CrossAttn | 10.7 G       |
| BERT  | LoRA             | 8.6 G        |
| VIT   | LoRA             | 5.0 G        |

Results from distributed training may be 3-5 points lower than those from single-card training. This could be related to training scheduling, and it may be necessary to implement optimizer scheduling strategies for learning rates and other parameters.
