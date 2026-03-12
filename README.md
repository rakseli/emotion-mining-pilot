# emotion-mining-pilot

- How to run:
```bash

#!/bin/bash

set -euo pipefail
WORKDIR="$(pwd)"
TMPDIR=$WORKDIR
export PYTHONWARNINGS=ignore
#vllm and torch
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_TRITON_FLASH_ATTN=0
export VLLM_LOG_LEVEL=DEBUG
export VLLM_CACHE_ROOT="$TMPDIR/vllm_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR
export TRITON_CACHE_DIR=$TMPDIR
#export NCCL_DEBUG=INFO
#DISTRIBUTED
export OMP_NUM_THREADS=1
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=9999
python ../src/annotation/run_vllm.py --test --model_path $1

```