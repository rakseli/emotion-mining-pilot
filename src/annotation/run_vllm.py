import time
_START_TIME = time.time()
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

logger = logging.getLogger(__name__)


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

generic_system_prompt="""You are a helpful and focused AI assistant.

Always follow the user’s instructions carefully and complete the requested tasks to the best of your ability.
Provide clear, accurate, and relevant responses that stay on topic.
Do not include extra information or content that was not requested by the user.

"""

prompt_template = """
# Introduction
Input is a clinical note that can contain 0 or up to 34 distinct emotions.

The emotions are classified as follows:
1. Admiration: Admiration is a feeling of great liking and respect for a person or thing. 
2. Adoration: Deep respect or affection; fervent admiration or love.
3. Aesthetic Appreciation: The experience of beauty.
4. Amusement: Humour excited by something comical or funny; entertainment or enjoyment derived from this.
5. Anger: A strong feeling of displeasure, dissatisfaction, or annoyance, generally combined with antagonism or hostility towards a particular cause or object; the state of experiencing such feelings; wrath, rage, fury.
6. Anxiety: Worry over the future or about something with an uncertain outcome; uneasy concern about a person, situation, etc.; a troubled state of mind arising from such worry or concern.
7. Awe: The feeling of respect and amazement that you have when you are faced with something wonderful and often rather frightening.
8. Awkwardness: Lack of skill or dexterity; clumsiness.
9. Boredom: The state of being bored.
10. Calmness: The state or quality of being calm; stillness, tranquillity, quietness. Freedom from agitation or disturbance.
11. Confusion: The confounding or mistaking of one for another; failure to distinguish. Const. of (things), of one with another, between (things). It is not clear what the true situation is, especially because people believe different things.
12. Contempt: A feeling of dislike or hostility towards a person or thing one regards as inferior, worthless, or despicable; an attitude expressive of such a feeling; (later) a complete lack of consideration or respect for a person or thing.
13. Craving: An intense desire or longing.
14. Disappointment: Deprivation or denial of something required, desired, or expected; spec. failure in the proper equipping of a store, or in the expected provision of goods, supplies, etc.
15. Disgust: Strong repugnance, aversion, or repulsion excited by that which is loathsome or offensive, as a foul smell, disagreeable person or action, disappointed ambition, etc.; profound instinctive dislike or dissatisfaction.
16. Empathic Pain: Have an experience of another's pain. Being able to understand what others feel, be it an emotion or a sensory state.
17. Entrancement: The state of being filled with wonder and delight; enchantment. The condition of being put into a trance; hypnotization.
18. Envy: The feeling you have when you wish you could have the same thing or quality that someone else has.
19. Excitement: A person or thing that excites; stimulation or thrill
20. Fear: The emotion of pain or uneasiness caused by the sense of impending danger, or by the prospect of some possible evil.
21. Guilt: An unpleasant feeling of having committed wrong or failed in an obligation; a guilty feeling.
22. Horror: A feeling of great shock, fear, and worry caused by something extremely unpleasant
23. Interest: A feeling of particular concern for or curiosity about a person or thing; attention or consideration devoted to a person or thing; engagement with a subject, topic, etc. 
24. Joy: A vivid emotion of pleasure arising from a sense of well-being or satisfaction; the feeling or state of being highly pleased or delighted; exultation of spirit; gladness, delight.
25. Nostalgia: Is an affectionate feeling you have for the past, especially for a particularly happy time.
26. Pride: A feeling of satisfaction that you have because you or people close to you have done something good or possess something good.
27. Relief: Alleviation of or deliverance from distress, anxiety, or some other emotional burden; the feeling accompanying this; mental relaxation, release, or reassurance.
28. Romance: Ardour or warmth of feeling in a love affair; love, esp. of an idealized or sentimental kind.
29. Sadness: Feel unhappy, usually because something has happened that you do not like.
30. Satisfaction: The state or quality of feeling satisfied or contented; (in later use chiefly) gratification, pleasure, or contentment caused by a fact, event, or state of things.
31. Sexual Desire: A wish, need, or drive to seek out sexual objects or to engage in sexual activities.
32. Surprise: To affect with the characteristic emotion caused by something unexpected; to excite to wonder by being unlooked-for.
33. Sympathy: The sharing of another's emotions, esp of sorrow or anguish; pity; compassion.
34. Triumph: The feeling of exultation and happiness derived from a victory or major achievement.

# Instructions
Extract the patient's emotions that the healthcare professional may have documented in the note. Use the categories given above.
The note is given after the header "# Note".
Be careful to extract only the patient's emotions, not those of the professional or a close relative.
Verify your decision by applying logical principles. Justify your decision with a maximum of three sentences.
Sometimes, a note may not contain any emotions. Answer then "No emotions". Notes may contain long lists of measurements, and they do not change the instruction.
After the extraction, output a confidence score from 0.0 to 1.0 for the answer. Confidence 0.0 means the answer is unreliable, 0.5 means the answer is somewhat confident, and 1 means there is no possible error.
## Output format
Output both justification and answer in the following format:

**Justification:** justification for the answer

**Confidence:** a score from 0.0 to 1.0

**Answer:** emotion_1, emotion_2


DO NOT PROVIDE REASONING OR ANY ADDITIONAL INFORMATION IN THE ANSWER OR CONFIDENCE SECTIONS!
# Note
{note}
"""

SEED = 42


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

        
def create_dataloader(args):
    num_workers = int(os.getenv("SLURM_CPUS_PER_TASK",1))
    num_workers = num_workers-1 
    if not os.path.isdir(args.root_path):
        data_files = [args.root_path]
    else:
        data_files = glob.glob(os.path.join(args.root_path,"*.json"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dataset = IterableDataset.from_generator(lambda: data_generator(data_files))
    dataset = dataset.map(format_data,fn_kwargs={'tokenizer':tokenizer,'args':args})
    dataloader = DataLoader(dataset,batch_size=200,collate_fn=naive_data_collator,shuffle=False,num_workers=num_workers)
    return dataloader

def format_data(example,tokenizer,args):
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
    parser.add_argument("--root_path",default="/scratch/project_2017000/emotion-mining-pilot/data/generated_llama3.1-8b_fewshot_sample_ids.jsonl",type=str,help="path to source file")
    parser.add_argument("--output_path",type=str,default="/scratch/project_2017000/emotion-mining-pilot/results")
    parser.add_argument("--exit_duration_in_mins",type=int, default=None, help="exit duration")
    parser.add_argument("--test",action="store_true")
    args = parser.parse_args()
    model_name = os.path.basename(args.model_path)
    dataloader = create_dataloader(args)
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
    with open(os.path.join(args.output_path,f"{model_name}_testing_results.jsonl"),"w") as fi:
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
