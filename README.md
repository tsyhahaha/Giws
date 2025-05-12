# GIW  
**GIW (Give It a Whirl)** – Best practices for PyTorch training from scratch.

All models, training, and related configurations in this project are based on **Hydra**, a powerful tool for managing and organizing configuration parameters via `yaml` files. The basic format can be found under the `config/` directory. For advanced usage, refer to the [official Hydra documentation](https://hydra.cc/docs/intro/).

## Getting Started

```bash
git clone git@github.com:tsyhahaha/Giws.git
cd Giws
pip install -e .
```

Then, configure the training parameters in `config/xxx.yaml`, and update the config filename in `run_train.sh` accordingly:

```bash
torchrun \
    --nnodes=1 \
    --master-port 29505 \
    --nproc_per_node=1 \
    ./train.py \
    --config-name xxx(.yaml)
```

Now you can start training with the specified configuration file:

```bash
bash run_train.sh
```

## Projects

This repository includes implementations of the following projects:
* Handwritten digit recognition on MNIST using CNN
* Image classification on CIFAR10 using ViT
* Chinese poem generation using LSTM
* Chinese-English translation using Transformer

## TODO list
- [ ] imgs about training results of each project in `docs/`
- [ ] details of environment build methods(both pipenv/conda)
- [ ] more training-friendly project structure and code organization
