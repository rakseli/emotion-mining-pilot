import json
import os
import json
import argparse
import glob
import pandas as pd

def normalize_text(s: str) -> str:
    """
    Normalize text by:
      1) removing all carriage returns ('\r')
      2) collapsing double newlines into a single newline (repeatedly)
    """
    if s is None:
        return s
    s = s.replace("\r", "")
    while "\n\n" in s:
        s = s.replace("\n\n", "\n")
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root",default="",type=str)
    parser.add_argument("--use_strucural_data",action="store_true")
    args = parser.parse_args()
    file_paths = glob.glob(os.path.join(args.input_root,"*.csv"))
    example_notes = []
    for f_p in file_paths:
        df = pd.read_csv(f_p)
        p_id = os.path.basename(f_p).split(".")[0]
        p_dict = {"id":p_id}
        t_items = []
        for row in df.iterrows():
            row_text = ""
            if isinstance(row[1]['Date'],str):
                if not args.use_strucural_data:
                    add_other_rows = False
                    if isinstance(row[1]['Heading (narrative)'],str):
                        add_other_rows = True
                        if isinstance(row[1]['Time'],str):
                            row_text = row_text + f"{row[1]['Date']} {row[1]['Time']} "
                        else:
                            row_text = row_text + row[1]['Date'] + ": "
                else:
                    if isinstance(row[1]['Time'],str):
                        row_text = row_text + f"{row[1]['Date']} {row[1]['Time']} "
                    else:
                        row_text = row_text + row[1]['Date'] + " "
            if isinstance(row[1]['Heading (rakenteinen)'],str):
                if not args.use_strucural_data:
                    if add_other_rows:
                        if isinstance(row[1]['Value'],str):
                            row_text = row_text +f"{row[1]['Heading (rakenteinen)']}: {row[1]['Value']} "
                        else:
                            row_text = row_text + row[1]['Heading (rakenteinen)']+" "
            
                else:
                    if isinstance(row[1]['Value'],str):
                        row_text = row_text +f"{row[1]['Heading (rakenteinen)']}: {row[1]['Value']} "
                    else:
                        row_text = row_text + row[1]['Heading (rakenteinen)']+": "
            if isinstance(row[1]['Heading (narrative)'],str):
                row_text+=row[1]['Heading (narrative)']+": "
            if isinstance(row[1]['Free text'],str):
                row_text+=row[1]['Free text']
            if row_text != '':
                t_items.append(row_text)
        p_dict['text']=normalize_text("\n".join(t_items))
        example_notes.append(p_dict)
    
    with open(os.path.join(args.input_root,f"example_notes_with_structural_data_{str(args.use_strucural_data)}.jsonl"),"w") as o_f:
        for e in example_notes:
            j_l = json.dumps(e,ensure_ascii=False)
            o_f.write(j_l+"\n")
            