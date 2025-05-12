torchrun \
    --nnodes=1 \
    --master-port 29505 \
    --nproc_per_node=1 \
    ./train.py hydra.output_subdir=null \
        --config-name train_transformer.yaml
