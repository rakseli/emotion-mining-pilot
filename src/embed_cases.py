import argparse
import json
import os
import pickle
import torch
import numpy as np
import sys
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
from typing import Dict, Iterable, List, Tuple


def read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}: {e}") from e


def clean_text(text: str) -> str:
    # Requirement: remove the substring "# Generated EHR"
    return text.replace("# Generated Progress notes", "")


def split_cases(text: str) -> List[str]:
    # Split on double newlines and keep only non-empty, stripped candidates
    candidates = [c.strip() for c in text.split("\n\n")]
    candidates = [c for c in candidates if c]
    candidates = [c for c in candidates if len(c)<3000 and len(c)>200]
    cases = [c.replace("\n\n", "\n") for c in candidates]

    # Only accept the split if every candidate has at least 2 lines
    if all(len(c.splitlines()) >= 2 for c in cases):
        return cases

    # Otherwise return the original text unchanged (as a single-element list)
    if len(text)>200 and len(text)<3000:
        cases = [text.replace("\n\n", "\n")]
    else:
        cases = None
    
    return cases


def make_case_id(source_file: str, record_index: int, case_index: int) -> str:
    base = os.path.basename(source_file)[:-6]
    return f"{base}_rec{record_index}_case{case_index}"


def farthest_point_sampling(
    cases: list[tuple],
    z: int,
    metric: str = 'cosine',
    outlier_pct: float = 0.10,
) -> set:
    """
    Select z maximally diverse cases
    after discarding the top `outlier_pct` most distant cases from the centroid.

    Args:
        cases: list of (case_id, embedding) tuples
        z: number of cases to select
        metric: distance metric ('euclidean', 'cosine', etc.)
        outlier_pct: fraction of farthest-from-centroid cases to exclude

    Returns:
        Set of selected case_ids
    """
    case_ids, embeddings = zip(*cases)
    case_ids = list(case_ids)
    vectors = np.array(embeddings)
    n = vectors.shape[0]

    centroid = vectors.mean(axis=0, keepdims=True)
    dists_to_centroid = cdist(vectors, centroid, metric=metric).flatten()

    cutoff = np.percentile(dists_to_centroid, (1 - outlier_pct) * 100)
    keep_mask = dists_to_centroid <= cutoff

    kept_indices = np.where(keep_mask)[0]
    vectors = vectors[kept_indices]
    case_ids = [case_ids[i] for i in kept_indices]
    n = vectors.shape[0]

    assert z <= n, (
        f"After removing top {outlier_pct:.0%} outliers, only {n} cases "
        f"remain but z={z} were requested."
    )

    # Seed with the vector closest to centroid (recompute on filtered set)
    centroid = vectors.mean(axis=0, keepdims=True)
    dists_to_centroid = cdist(vectors, centroid, metric=metric).flatten()
    first_idx = np.argmin(dists_to_centroid)

    selected_indices = [first_idx]

    min_dists = cdist(vectors, vectors[first_idx:first_idx + 1], metric=metric).flatten()
    min_dists[first_idx] = -np.inf

    for _ in range(z - 1):
        next_idx = np.argmax(min_dists)
        selected_indices.append(next_idx)
        new_dists = cdist(vectors, vectors[next_idx:next_idx + 1], metric=metric).flatten()
        min_dists = np.minimum(min_dists, new_dists)
        min_dists[next_idx] = -np.inf

    return {case_ids[i] for i in selected_indices}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_path",
        required=True,
        help="Path to a .jsonl file or a directory containing .jsonl file.",
    )
    ap.add_argument(
        "--id_field",
        default="id",
    )
    ap.add_argument(
        "--model",
        default="TurkuNLP/finnish-modernbert-large-short-5e-05-msmarco",
        help="SentenceTransformers encoder model name or local path.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding computation.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="whether to recompute",
    )
    args = ap.parse_args()
    
    output_embeddings_path =  os.path.join(os.path.dirname(args.input_path),f"{os.path.basename(args.input_path)[:-6]}_embeddings.pkl")
    output_texts_path = os.path.join(os.path.dirname(args.input_path),f"{os.path.basename(args.input_path)[:-6]}_all_texts.jsonl")
    output_selected_text_path = os.path.join(os.path.dirname(args.input_path),f"{os.path.basename(args.input_path)[:-6]}_different_texts.jsonl")
    should_compute = True
    if os.path.exists(output_embeddings_path):
        print(f"Embeddings already computed ({output_embeddings_path})")
        if not args.force:
            should_compute = False
            
    if should_compute:
        case_ids: List[str] = []
        case_texts: List[str] = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cpu":
            raise ValueError("Device should be GPU for efficient computing!")
        print("Loading model",flush=True)
        model_kwargs = {"attn_implementation": "flash_attention_2", "torch_dtype": "bfloat16"}
        model = SentenceTransformer(args.model,model_kwargs=model_kwargs, device=device)
        model.max_seq_length=4096
        print("Loading data",flush=True)
        for rec_i, obj in enumerate(read_jsonl(args.input_path)):
            raw = obj["generated_text"]
            if raw is None:
                continue
            text = clean_text(str(raw))
            cases = split_cases(text)
            if cases is None:
                continue
            
            for case_i, case in enumerate(cases):
                cid = make_case_id(args.input_path, rec_i, case_i)
                case_ids.append(cid)
                case_texts.append(case)

        if not case_texts:
            raise ValueError("No cases found to embed (check input files and 'generated_text' field).")

        # Embed in batches
        print("Embedding data",flush=True)
        embeddings = model.encode(
            case_texts,
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=True,
        ).astype(np.float16)
        print("Done embedding",flush=True)
        case_embeddings: List[Tuple[str, np.ndarray]] = list(zip(case_ids,embeddings))

        os.makedirs(os.path.dirname(os.path.abspath(output_embeddings_path)), exist_ok=True)
        with open(output_embeddings_path, "wb") as f:
            pickle.dump(case_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(output_texts_path,"w") as f:
            for t,i in zip(case_texts,case_ids):
                j_l = json.dumps({"text":t,"id":i},ensure_ascii=False)
                f.write(j_l+ '\n')
        case_texts = [{'text':t,'id':i} for t,i in zip(case_texts,case_ids)]
    else:
        with open(output_embeddings_path, 'rb') as f:
            case_embeddings = pickle.load(f)
        
        with open(output_texts_path) as f:
            case_texts = []
            for line in f:
                try:
                    case_texts.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in {output_texts_path}:{e}") from e
    
    selected_ids = farthest_point_sampling(case_embeddings, z=1000, metric='cosine')
    with open(output_selected_text_path,"w") as f:
        for c in case_texts:
            if c['id'] in selected_ids:
                j_l = json.dumps({"text":c['text'],"id":c['id'] },ensure_ascii=False)
                f.write(j_l+ '\n')

if __name__ == "__main__":
    main()