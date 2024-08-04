# !/bin/bash
#set -x


IMAGENET
ln -s /lib64/libcuda.so.1 /lib64/libcuda.so
PYTHONPATH=/home/tsternal/wf/llamagen python3 /home/tsternal/wf/llamagen/tokenizer/tokenizer_image/vq_train.py \
 --data-path="/home/tsternal/wf/data/imagenet/ILSVRC/Data/CLS-LOC/train" \
 --image-size=256 --disc-type="patchgan" \
 --cloud-save-path="/home/tsternal/wf/llamagen/outputs/llamagen_test_output" --num-workers=1 --global-batch-size=1 \
 --gradient-accumulation-steps=8 --vq-model="VQ-VAE-VAR" \
 --enable-wandb --wandb-key=$WANDB_KEY \
  "$@"
