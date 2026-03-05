import os
import json
import argparse
from openai import OpenAI
from run_vllm import prompt_template, data_generator,naive_data_collator,extract_text, EMOTIONS
from datasets import IterableDataset,disable_caching
from torch.utils.data import DataLoader, get_worker_info

def create_dataloader(args):
    num_workers = int(os.getenv("SLURM_CPUS_PER_TASK",1))
    num_workers = num_workers-1 
    if not os.path.isdir(args.root_path):
        data_files = [args.root_path]
    else:
        data_files = glob.glob(os.path.join(args.root_path,"*.json"))
    dataset = IterableDataset.from_generator(lambda: data_generator(data_files))
    dataloader = DataLoader(dataset,batch_size=200,collate_fn=naive_data_collator,shuffle=False,num_workers=num_workers)
    return dataloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path",default="/scratch/project_2017000/emotion-mining-pilot/data/generated_llama3.1-8b_fewshot_sample_ids.jsonl",type=str,help="path to source file")
    parser.add_argument("--output_path",type=str,default="/scratch/project_2017000/emotion-mining-pilot/results")
    parser.add_argument("--test",action="store_true")
    args = parser.parse_args()
    if args.test:
        model_name = "gpt-5.2-2025-12-11"
        #model_name = "gpt-5-nano-2025-08-07"
    else:
        model_name = "gpt-5-nano-2025-08-07"
        
    dataloader = create_dataloader(args)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    with open(os.path.join(args.output_path,f"{model_name}_testing_results.jsonl"),"w") as fi:
        for batch_index,batch in enumerate(dataloader,start=1):
            print(f"Running batch {batch_index}")
            for t in batch:
                prompt = prompt_template.format(note=t['text'])
                response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                )
                generated_text = response.choices[0].message.content
                print(f"Prompt: {prompt[-100:]!r}..., Generated text: {generated_text[:100]!r}...")
                d = {"model":model_name,"prompt":prompt,"generated_text":generated_text,"id":t['id']}
                j_a = extract_text(generated_text)
                d.update(j_a)
                json_line = json.dumps(d,ensure_ascii=False)
                fi.write(json_line + '\n')
            if args.test:
                if batch_index==1:
                    break
