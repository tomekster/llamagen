#!/bin/bash

#SBATCH --job-name=train_vq    # create a short name for your job
#SBATCH --partition=normal    # use the GPU partition
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4      # use torchrun to handle local ranks
##SBATCH --gpus-per-node=4            # number of gpus per node
#SBATCH -c 18                # total cores 72 (=18 * 4),cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=0               # total memory per node 
#SBATCH --exclusive
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=logs/train_vq_dist_%j.out  # output file


##### Number of total processes 
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

# If you want to load things from your .bashrc profile, e.g. cuda drivers, singularity etc 
#source ~/.bashrc


# ******************* These are read internally it seems ***********************************
# ******** Master port, address and world size MUST be passed as variables for DDP to work 
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export RANK=$SLURM_PROCID
export NODE_RANK=$SLURM_NODEID
export LOCAL_RANK=$SLURM_LOCALID
export LOCAL_SIZE=$SLURM_NTASKS_PER_NODE
echo "MASTER_PORT"=$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE
echo "LOCAL_RANK="$LOCAL_RANK
echo "RANK="$RANK

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
# ******************************************************************************************

export OMP_NUM_THREADS=4
# zoom zoom - recommended from lightning
#export NCCL_NSOCKS_PERTHREAD=4
#export NCCL_SOCKET_NTHREADS=2
#export NCCL_MIN_CHANNELS=32


echo "Run started at:- "
date

#export PYTHONPATH=/workspace/LlamaGen
export PYTHONPATH=/workspace/llamagen
export http_proxy=http://proxy.cscs.ch:8080 
export https_proxy=http://proxy.cscs.ch:8080

#srun --environment=torchcontainer torchrun \
#srun sarus run --mount type=bind,source=/users/lhuang/FoundationModel,target=/workspace --mpi \
#    -e TORCH_CPP_LOG_LEVEL=INFO \
#    -e TORCH_DISTRIBUTED_DEBUG=DETAIL \
#    ghcr.io/huanglangwen/foundationmodel \
#srun --mpi=pmi2 --environment=torchcontainer bash -c "pip install hanging_threads && torchrun \
#srun --mpi=pmi2 --environment=torchcontainer torchrun \
#    --nnodes $SLURM_JOB_NUM_NODES \
#    --nproc_per_node $SLURM_NTASKS_PER_NODE \
#    --rdzv_id $RANDOM \
#    --rdzv_backend c10d \
#    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
#    /workspace/FoundationModel/train/graphcast/train_graphcast.py
# global batch size: 8 (node) * 4 (gpu) * 3 (batch per gpu) = 96
# disc start: 6000 (batch size is 3 times larger than default)

 # DISTRIBUTED
srun --mpi=pmi2 --environment=atmtokenizer -o report_vq_$SLURM_JOBID -- \
  python3 llamagen/tokenizer/tokenizer_image/vq_train.py \
 --dataset=imagenet --data-path="/home/tsternal/imagenet/train" \
 --image-size=256 --disc-type="patchgan" --epochs=4 \
 --cloud-save-path="/outputs/" --num-workers=4 --global-batch-size=96 \
 --gradient-accumulation-steps=1 --ema --vq-model="VQ-VAE-VAR" --disc-start=20000 \
 --enable-wandb --wandb-key=$WANDB_KEY --interpolate-data
