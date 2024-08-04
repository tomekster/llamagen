# !/bin/bash
set -x

torchrun \
--nnodes=$nnodes --nproc_per_node=1 --node_rank=1 \
--master_addr=$master_addr --master_port=8080 \
tokenizer/tokenizer_image/vq_train.py "$@"