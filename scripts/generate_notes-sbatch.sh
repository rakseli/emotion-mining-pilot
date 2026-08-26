#!/bin/bash
#SBATCH --job-name=production_generate_notes
#SBATCH --account=<project_name>
#SBATCH --time=05:00:00
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=gpumedium
#SBATCH --output=../logs/generation/%x_%j.output
#SBATCH --error=../logs/generation/%x_%j.error
echo "Start $(date +"%Y-%-m-%d-%H:%M:%S")"
module purge
set -euo pipefail
#Mahti
CONTAINER_PATH="/scratch/<project_name>/containers/pytorch_container_20260407_205334.sif"
export LOCAL_SCRATCH=/scratch/project_2018556/.cache
TMPDIR=$LOCAL_SCRATCH
echo "Cache location: $TMPDIR"
#Python
export PYTHONWARNINGS=ignore
#HF
export HF_HOME=/scratch/project_2018556/.cache/hf_cache
#vllm and torch
export VLLM_CONFIGURE_LOGGING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_CACHE_ROOT="$TMPDIR/vllm_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor_cache"
export TRITON_CACHE_DIR="$TMPDIR/triton_cache"
export APPTAINERENV_LD_PRELOAD=/appl/soft/ai/lib/fake_tcp_ulp.so
#DISTRIBUTED
export OMP_NUM_THREADS=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
apptainer exec --bind="/users,/appl/soft/ai/lib/,/projappl,/scratch,$LOCAL_SCRATCH" --nv $CONTAINER_PATH python ../src/generate_notes.py --exit_duration_in_mins 295 \
 --run_name short_prompt_short_examples_production \
 --example_path /scratch/<project_name>/emotion-mining-pilot/data/example_notes_short.jsonl

echo "End $(date +"%Y-%-m-%d-%H:%M:%S")" 