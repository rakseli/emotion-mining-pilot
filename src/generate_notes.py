import time
_GENERATION_START_TIME = time.time()
import glob
import os
import json
import argparse
import torch
import sys
import re
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
import numpy as np
from datasets import Dataset
from torch.utils.data import DataLoader, get_worker_info
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationMode
from prompt_tools import build_prompt_with_sampling, generic_system_prompt
from annotate_emotions import naive_data_collator
from datetime import datetime

logger = logging.getLogger(__name__)

def generate_dataset(n:int,example_path:str) -> list:
    prompts = []
    for i in range(1,n):
        prompt = build_prompt_with_sampling(seed=np.random.randint(i+100),example_path=example_path)
        prompts.append({'text':prompt,'id':i})
    dataset = Dataset.from_list(prompts)
    return dataset

def format_data(example,tokenizer):
    user = {"role": "user", "content":example['text']}
    system = {"role":"system","content":generic_system_prompt}
    example['text'] = tokenizer.apply_chat_template([system,user],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    return example

def create_dataloader(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dataset = generate_dataset(n=args.n_notes,example_path=args.example_path)
    dataset = dataset.map(format_data,fn_kwargs={'tokenizer':tokenizer})
    dataloader = DataLoader(dataset,batch_size=100,collate_fn=naive_data_collator,shuffle=False)
    return dataloader

def count_lines(path):
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n

def detect_completed_batches(out_path, batch_size):
    """
    Returns number of *completed* batches already present in jsonl.
    We assume you write exactly one json line per sample.
    Only full batches are considered completed; a partial last batch will be re-run.
    """
    if not os.path.exists(out_path):
        return 0
    n_lines = count_lines(out_path)
    return n_lines // batch_size

def skip_batches(dataloader, n_to_skip):
    """
    Advance the iterator by n_to_skip batches.
    """
    if n_to_skip == 0:
        return dataloader

    it = iter(dataloader)
    for _ in range(n_to_skip):
        next(it)
    return it


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",type=str,default="/scratch/project_2018556/models/Qwen3.5-122B-A10B-GPTQ-Int4")
    parser.add_argument("--output_path",type=str,default="/scratch/project_2017000/emotion-mining-pilot/results")
    parser.add_argument("--n_notes",default=5000,help="number of prompts to generate")
    parser.add_argument("--run_name",default=None)
    parser.add_argument("--example_path",default=None)
    parser.add_argument("--exit_duration_in_mins",type=int, default=None, help="exit duration")
    parser.add_argument("--test",action="store_true")
    args = parser.parse_args()
    if args.test:
        args.n_notes = 1000
    SEED = 66
    model_name = os.path.basename(args.model_path)
    dataloader = create_dataloader(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    eos_id = tokenizer.eos_token_id
    example_prompt = build_prompt_with_sampling(seed=np.random.randint(1,100),example_path=args.example_path)
    prompt_len = len(tokenizer(example_prompt).input_ids)
    sys_len = len(tokenizer(generic_system_prompt).input_ids)
    logger.info(f"Prompt lenght in tokens: {prompt_len+sys_len}")
    del tokenizer
    del example_prompt
    temperature = 0.7
    top_p = 0.8
    top_k = 20
    min_p=0.0
    gpu_mem=0.9
    enforce_eager = False
    presence_penalty=1.5
    repetition_penalty=1.0
    batch_size = 16
    max_model_len = 72000
    tensor_parallel_size = 4
    llm = LLM(model=args.model_path,tensor_parallel_size=tensor_parallel_size,
              max_num_seqs=batch_size,distributed_executor_backend="mp",
              disable_custom_all_reduce=True,
              max_model_len=max_model_len,gpu_memory_utilization=gpu_mem,
              enable_chunked_prefill=True,
              enforce_eager=enforce_eager,
              language_model_only=True
              )

    sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=4096,
                seed=SEED,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                stop_token_ids=[eos_id]
                )
    total_tokens = 0
    start = time.time()
    proccessed_prompts = 0
    logger.info(f"Starting to process data")
    if args.run_name is None:
        if args.test:
            output_suffix = "test_run"
        else:
            output_suffix = "production_run"
    else:
        output_suffix = args.run_name

    # Determine where to start (batch index is 1-based)
    out_path = os.path.join(args.output_path,f"{model_name}_{output_suffix}_note_generation_results.jsonl")
    batches_to_skip = 0
    # Auto-detect based on existing file length
    dataloader_batch_size = dataloader.batch_size
    batches_to_skip = detect_completed_batches(out_path, dataloader_batch_size)
    logger.info(f"Output: {out_path}")
    logger.info(f"Resuming: skip {batches_to_skip} batches")
    # Open file in append mode if resuming and file exists, else write mode
    file_mode = "a" if (batches_to_skip > 0 and os.path.exists(out_path)) else "w"
    start = time.time()
    total_tokens = 0
    proccessed_prompts = 0
    logger.info("Starting to process data")
    all_generated = True
    with open(out_path, file_mode, encoding="utf-8") as fi:
        # Create an iterator and skip batches
        dl_iter = skip_batches(dataloader, batches_to_skip)
        for batch_index,batch in enumerate(dl_iter,start=1):
            elapsed_time = (time.time() - _GENERATION_START_TIME) / 60.0
            if elapsed_time > args.exit_duration_in_mins:
                logger.info('Exiting program gracefylly after {} minutes'.format(elapsed_time),flush=True)
                all_generated = False
                break
            texts = [text_i['text'] for text_i in batch]
            ids = [s['id'] for s in batch]
            logger.info(f"Batch has {len(texts)} texts")
            logger.info(f"Running batch {batch_index}")
            outputs = llm.generate(texts, sampling_params)
            logger.info(f"Done batch {batch_index}")
            # compute throughput
            b_tokens = sum([len(o.outputs[0].token_ids) for o in outputs])
            total_tokens+=b_tokens
            for output,text_id in zip(outputs,ids):
                prompt = output.prompt
                generated_text = output.outputs[0].text
                logger.info(f"Prompt: {prompt[-100:]!r}..., Generated text: {generated_text[:100]!r}...")
                d = {"model":model_name,"prompt":prompt,"generated_text":generated_text,'id':text_id}
                json_line = json.dumps(d,ensure_ascii=False)
                fi.write(json_line + '\n')
                
            proccessed_prompts+=len(texts)
            if args.test:
                if batch_index==1:
                    logger.debug(f"Exiting the loop as args.test = {args.test}")
                    break
    elapsed = time.time() - start
    
    logger.info(f"Total prompts processed: {proccessed_prompts}")
    logger.info(f"Total tokens generated: {total_tokens}")
    logger.info(f"Tokens throughput: {total_tokens / elapsed:.2f} tokens/s")
    logger.info(f"Prompts throughput: {proccessed_prompts / elapsed:.2f} prompts/s")
    logger.info(f"Elapsed time: {elapsed:.2f}s ({elapsed/60:.2f} minutes)")
    if all_generated:
        logger.info("All questions were generated")
    else:
        logger.info("Generation did not exhaust")