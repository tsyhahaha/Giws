# GIW
GIW（Give It a Whirl）- Give It a Whirl: Best practices for pytorch training from scratch.

项目中的所有模型、训练等配置基于**hydra**, 该工具能够方便的将各种配置参数储存为 `yaml` 文件，基本格式参考 `config/` 下的配置文件，高级用法请参考[hydra官方文档](https://hydra.cc/docs/intro/).

## 开始

```
git@github.com:tsyhahaha/Giws.git
cd Giws
pip install -e .
```

然后在 `config/xxx.yaml` 中，配置训练所需参数，将 `run_train.sh` 中的配置文件名改为对应文件名：

```
torchrun \
    --nnodes=1 \
    --master-port 29505 \
    --nproc_per_node=1 \
    ./train.py \
    --config-name xxx(.yaml)
```

即可通过指定配置文件启动训练：

```
bash run_train.py
```

## 项目

本仓库实现的项目包括：
* 基于 CNN 的 MINST 手写数字图片识别
* 基于 ViT 的 CIFAR10 图片分类
* 基于 LSTM 的中文古诗生成
* 基于 Transformer 的中英文翻译

