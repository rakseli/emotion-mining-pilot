#!/bin/bash
#SBATCH --job-name=vllm
#SBATCH --account=<project_name>
#SBATCH --time=02:00:00
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=gpumedium
#SBATCH --output=../logs/annotation/%x_%j.output
#SBATCH --error=../logs/annotation/%x_%j.error
echo "Start $(date +"%Y-%-m-%d-%H:%M:%S")"
module purge
module load pytorch/2.9
set -euo pipefail
#Mahti
export LOCAL_SCRATCH=/scratch/<project_name>/.cache
TMPDIR=$LOCAL_SCRATCH
echo "Cache location: $TMPDIR"
#Python
export PYTHONWARNINGS=ignore
#vllm and torch
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_TRITON_FLASH_ATTN=0
export VLLM_LOG_LEVEL=DEBUG
export VLLM_CACHE_ROOT="$TMPDIR/vllm_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor_cache"
export TRITON_CACHE_DIR="$TMPDIR/triton_cache"
#export NCCL_DEBUG=INFO
#DISTRIBUTED
export OMP_NUM_THREADS=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
srun python ../src/annotation/run_vllm.py --test --model_path $1
echo "End $(date +"%Y-%-m-%d-%H:%M:%S")"