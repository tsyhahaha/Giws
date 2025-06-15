torchrun \
    --nnodes=1 \
    --master-port 29506 \
    --nproc_per_node=2 \
    ./train.py \
    --config-name train_transformer