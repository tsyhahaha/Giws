torchrun \
    --nnodes=1 \
    --master-port 29504 \
    --nproc_per_node=4 \
    ./train.py hydra.output_subdir=null \
        --config-name train_transformer.yaml
