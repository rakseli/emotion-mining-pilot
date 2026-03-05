import json
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",type=str)
    args = parser.parse_args()
    
    with open(args.input_file,"r") as r_f, open(args.input_file.split(".")[0]+".jsonl","w") as w_f:
        for l in r_f:
            d = {'text':l}
            json_line = json.dumps(d,ensure_ascii=False)
            w_f.write(json_line + '\n')