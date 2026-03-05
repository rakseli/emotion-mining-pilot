import json
import argparse
import uuid

if __name__ == "__main__":
    # Add id field
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",default="/scratch/project_2017000/emotion-mining-pilot/data/generated_llama3.1-8b_fewshot_sample.jsonl",type=str)
    parser.add_argument("--output_path",type=str,default="/scratch/project_2017000/emotion-mining-pilot/data/generated_llama3.1-8b_fewshot_sample_ids.jsonl")
    args = parser.parse_args()

    with open(args.data_path,"r") as input_file, open(args.output_path,"w") as output_file:
        for i,l in enumerate(input_file,start=1):
            data = json.loads(l)
            data['id'] = str(uuid.uuid4())
            json_line = json.dumps(data,ensure_ascii=False)
            output_file.write(json_line+"\n")
            