import time
import glob
import os
import json
import argparse
import torch
import sys
import re
import logging
import numpy as np
from datasets import Dataset
from torch.utils.data import DataLoader, get_worker_info
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationMode
from prompt_tools import build_prompt_with_sampling
from annotate_emotions import naive_data_collator
logger = logging.getLogger(__name__)

def generate_dataset(n:int) -> list:
    prompts = []
    for i in range(1,n):
        prompt = build_prompt_with_sampling(seed=np.random.randint(i))
        prompts.append({'text':prompt,'id':i})
    dataset = Dataset.from_list(prompts)
    return dataset

def format_data(example,tokenizer):
    user = {"role": "user", "content":example['text']}
    example['text'] = tokenizer.apply_chat_template([user],tokenize=False,reasoning_effort="low")
    return example

def create_dataloader(args):
    num_workers = int(os.getenv("SLURM_CPUS_PER_TASK",1))
    num_workers = num_workers-1 
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dataset = generate_dataset(n=args.n_notes)
    dataset = dataset.map(format_data,fn_kwargs={'tokenizer':tokenizer})
    dataloader = DataLoader(dataset,batch_size=200,collate_fn=naive_data_collator,shuffle=False,num_workers=num_workers)
    return dataloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",default="/scratch/project_2017000/models/gpt-oss-120b",type=str)
    parser.add_argument("--input_path",default="/scratch/project_2017000/emotion-mining-pilot/data/generated_llama3.1-8b_fewshot_sample_ids.jsonl",type=str,help="path to source file")
    parser.add_argument("--output_path",type=str,default="/scratch/project_2017000/emotion-mining-pilot/results")
    parser.add_argument("--n_notes",default=10000,help="number of prompts to generate")
    parser.add_argument("--test",action="store_true")
    args = parser.parse_args()
    model_name = os.path.basename(args.model_path)
    dataloader = create_dataloader(args)
    for batch in dataloader:
        for s in batch:
            print(s)
        if args.test:
            break
    sys.exit(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    eos_id = tokenizer.eos_token_id
    del tokenizer
    temperature = 1
    top_p = 1
    top_k = 1
    quantization="mxfp4"
    gpu_mem=0.8
    enforce_eager = False
    batch_size = 4
    SEED = 66
    llm = LLM(model=args.model_path,tensor_parallel_size=4,
              max_num_seqs=batch_size,distributed_executor_backend="mp",
              disable_custom_all_reduce=True,
              max_model_len=42000,gpu_memory_utilization=gpu_mem,
              enable_chunked_prefill=True,quantization=quantization,
              enforce_eager=enforce_eager)
    
    sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=32000,
                seed=SEED,
                presence_penalty=0.8,
                stop_token_ids=[eos_id]
                )
    total_tokens = 0
    start = time.time()
    proccessed_prompts = 0
    logger.info(f"Starting to process data")
    output_suffix = os.path.basename(args.input_path).split(".")[0]
    if args.test:
        output_suffix = "test_run"
    with open(os.path.join(args.output_path,f"{model_name}_{output_suffix}_results.jsonl"),"w") as fi:
        for batch_index,batch in enumerate(dataloader,start=1):
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
                j_a = extract_text(generated_text)
                d.update(j_a)
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
