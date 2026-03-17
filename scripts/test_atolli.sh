set -euo pipefail
source $HOME/venvs/pytorch/bin/activate
WORKDIR="$(pwd)"
TMPDIR=$WORKDIR
export PYTHONWARNINGS=ignore
#vllm and torch
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_CACHE_ROOT="$TMPDIR/vllm_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torch_inductor_cache"
export TRITON_CACHE_DIR="$TMPDIR/triton_cache"
#export NCCL_DEBUG=INFO
#DISTRIBUTED
export OMP_NUM_THREADS=1
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=9999
python ../src/annotation/run_vllm.py \
    --test \
    --root_path $HOME/emotion-mining-pilot/data/synthetic_data_sample.jsonl \
    --output_path $HOME/emotion-mining-pilot/results \
    --model_path $HOME/models/openai/gpt-oss-20b