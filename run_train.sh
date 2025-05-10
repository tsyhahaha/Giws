export CUDA_VISIBLE_DEVICES=1
torchrun \
    --nnodes=1 \
    --master-port 29504 \
    --nproc_per_node=1 \
    ./train.py --config-name train_vit.yaml
