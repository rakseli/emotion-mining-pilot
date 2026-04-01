import time
import glob
import os
import json
import argparse
import torch
import sys
import re
import logging
from datasets import IterableDataset,disable_caching
from torch.utils.data import DataLoader, get_worker_info
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationMode
from prompt_tools import emotion_prompt_template
logger = logging.getLogger(__name__)

def naive_data_collator(batch):
    """Does nothing, only for dataloader to batch samples 
    and not to convert them to tensors
    
    batch (list): list of dicts 
    Returns:
        list: list of dicts
    """    
    return batch

def extract_text(text):
    """
    Extracts a single justification and answer pair from text.
    Assumes only one of each exists in the text.
    Strips extra whitespace and newlines.
    """
    pattern = re.compile(
        r"(?:\*\*Justification:\*\*|Justification:)\s*\n*([\s\S]*?)"
        r"(?:\*\*Confidence:\*\*|Confidence:)\s*\n*([\s\S]*?)"
        r"(?:\*\*Answer:\*\*|Answer:)\s*\n*([\s\S]*?)$",

        flags=re.MULTILINE
    )

    match = pattern.search(text)
    if not match:
        return {'justification': None, 'answer': None ,'confidence': None}

    justification = match.group(1).strip()
    confidence = match.group(2).strip()
    answer_inter = match.group(3).strip()

    if "no emotions" in answer_inter.lower():
        answer = "no emotions"
    else:
        answer = re.split('[,;]', answer_inter)
        answer = [re.sub(r'[^A-Za-z0-9\s]', '',a.strip().lower()) for a in answer if a.strip().lower() in EMOTIONS]
    
    try:
        c = float(confidence)
    except:
        c = None
    
    return {'justification': justification, 'answer': answer, 'confidence': c}

def data_generator(data_files):
    
    def read_shard(data_file):
        with open(data_file) as f:
            for l in f:
                yield json.loads(l)

    
    worker_info = get_worker_info()
    if worker_info is None:
        assigned_shards = data_files
    else:
        per_worker = len(data_files) // worker_info.num_workers
        remainder = len(data_files) % worker_info.num_workers
        start = worker_info.id * per_worker + min(worker_info.id, remainder)
        end = start + per_worker + (1 if worker_info.id < remainder else 0)
        assigned_shards = data_files[start:end]
    for shard in assigned_shards:
        for example in read_shard(shard):
            yield example

        
def create_dataloader(prompt_template,args):
    num_workers = int(os.getenv("SLURM_CPUS_PER_TASK",1))
    num_workers = num_workers-1 
    if not os.path.isdir(args.input_path):
        data_files = [args.input_path]
    else:
        data_files = glob.glob(os.path.join(args.input_path,"*.json"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dataset = IterableDataset.from_generator(lambda: data_generator(data_files))
    dataset = dataset.map(format_data,fn_kwargs={'tokenizer':tokenizer,"prompt_template":prompt_template,'args':args})
    dataloader = DataLoader(dataset,batch_size=200,collate_fn=naive_data_collator,shuffle=False,num_workers=num_workers)
    return dataloader

def format_data(example,tokenizer,prompt_template,args):
    user = {"role": "user", "content":prompt_template.format(note=example["text"])}
    if "gpt-oss-120b" in args.model_path:
        example['text'] = tokenizer.apply_chat_template([user],tokenize=False,reasoning_effort="low")
    else:
        system = {"role": "system", "content":generic_system_prompt}
        example['text'] = tokenizer.apply_chat_template([system,user],tokenize=False)
    return example

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",default="/scratch/project_2017000/models/gpt-oss-120b",type=str)
    parser.add_argument("--input_path",default="/scratch/project_2017000/emotion-mining-pilot/data/generated_llama3.1-8b_fewshot_sample_ids.jsonl",type=str,help="path to source file")
    parser.add_argument("--output_path",type=str,default="/scratch/project_2017000/emotion-mining-pilot/results")
    parser.add_argument("--test",action="store_true")
    args = parser.parse_args()
    # oxford dictionary: 2,3,5,6,7,8,9,10,12,13,14,15,18,20,21,23,24,27,28,30,32 
    # 4 https://www.tandfonline.com/doi/10.1080/02699930302297?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200pubmed
    # 16 https://www.science.org/doi/10.1126/science.1093535
    # Collins dictionary: 1,7,11,13,17,19,22,25,26,28,29,33
    # 31 Regan, P. C., & Berscheid, E. (1995). Gender differences in beliefs about the causes of male and female sexual desire. Personal Relationships, 2,345–358, p. 346
    EMOTIONS = {
    "admiration",
    "adoration",
    "aesthetic appreciation",
    "amusement",
    "anger",
    "anxiety",
    "awe",
    "awkwardness",
    "boredom",
    "calmness",
    "confusion",
    "contempt",
    "craving",
    "disappointment",
    "disgust",
    "empathic pain",
    "entrancement",
    "envy",
    "excitement",
    "fear",
    "guilt",
    "horror",
    "interest",
    "joy",
    "nostalgia",
    "pride",
    "relief",
    "romance",
    "sadness",
    "satisfaction",
    "sexual desire",
    "surprise",
    "sympathy",
    "triumph"
    }
    SEED = 42
    model_name = os.path.basename(args.model_path)
    dataloader = create_dataloader(prompt_template,args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    eos_id = tokenizer.eos_token_id
    logger.info(f"Prompt lenght: {len(tokenizer(prompt_template).input_ids)}")
    del tokenizer
    if "llama-3.3" in model_name:
        temperature = 0.6
        top_p = 0.9
        top_k = -1
        quantization="bitsandbytes"
        gpu_mem=0.5
        enforce_eager = True
        batch_size = 2
    elif "gpt-oss-120" in model_name:
        temperature = 1
        top_p = 1
        top_k = 1
        quantization="mxfp4"
        gpu_mem=0.8
        enforce_eager = False
        batch_size = 4
    else:
        raise ValueError(f"llama or gpt models should be used, {args.model_path} given")
    llm = LLM(model=args.model_path,tensor_parallel_size=4,
              max_num_seqs=batch_size,distributed_executor_backend="mp",
              disable_custom_all_reduce=True,
              max_model_len=32000,gpu_memory_utilization=gpu_mem,
              enable_chunked_prefill=True,quantization=quantization,
              enforce_eager=enforce_eager)
    
    sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=250,
                min_tokens=10,
                truncate_prompt_tokens=4096,
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
                logger.debug(f"Exiting the loop as args.test = {args.test}")
                break
    
    elapsed = time.time() - start
    
    logger.info(f"Total prompts processed: {proccessed_prompts}")
    logger.info(f"Total tokens generated: {total_tokens}")
    logger.info(f"Tokens throughput: {total_tokens / elapsed:.2f} tokens/s")
    logger.info(f"Prompts throughput: {proccessed_prompts / elapsed:.2f} prompts/s")
    logger.info(f"Elapsed time: {elapsed:.2f}s ({elapsed/60:.2f} minutes)")
