#!/bin/bash
#SBATCH --job-name=select_samples
#SBATCH --account=<project_name>
#SBATCH --time=00:10:00
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=32
##SBATCH --gres=gpu:a100:1
#SBATCH --partition=test
#SBATCH --output=../logs/generation/%x_%j.output
#SBATCH --error=../logs/generation/%x_%j.error
echo "Start $(date +"%Y-%-m-%d-%H:%M:%S")"
module purge
set -euo pipefail
#Mahti
CONTAINER_PATH="/scratch/<project_name>/containers/pytorch_container_mteb_20260423_221435.sif"
export LOCAL_SCRATCH=/scratch/project_2018556/.cache
TMPDIR=$LOCAL_SCRATCH
#HF
export HF_HOME=/scratch/project_2018556/.cache/hf_cache
#Apptainer
export APPTAINER_CACHEDIR=/scratch/<project_name>/.cache/apptainer

#torch
export APPTAINERENV_LD_PRELOAD=/appl/soft/ai/lib/fake_tcp_ulp.so
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor_cache"
export TRITON_CACHE_DIR="$TMPDIR/triton_cache"
apptainer exec --bind="/users,/appl/soft/ai/lib/,/projappl,/scratch,$LOCAL_SCRATCH" --nv $CONTAINER_PATH python ../src/embed_cases.py --input_path /scratch/<project_name>/emotion-mining-pilot/results/Qwen3.5-122B-A10B-GPTQ-Int4_short_prompt_short_examples_production_note_generation_results.jsonl
echo "End $(date +"%Y-%-m-%d-%H:%M:%S")"
