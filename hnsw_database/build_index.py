#!/usr/bin/env python3
import argparse, json, gzip
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


# ----------------------------------------------------
# Build clean text for embedding
# ----------------------------------------------------
def make_text(item):
    fields = ["title", "subtitle", "description", "features", "details"]
    chunks = []
    for k in fields:
        v = item.get(k)
        if not v:
            continue
        if isinstance(v, list):
            chunks.append(" ; ".join(str(x) for x in v))
        else:
            chunks.append(str(v))
    if not chunks:
        return None
    return " | ".join(chunks)


# ----------------------------------------------------
# Iterate through JSONL or JSONL.gz
# ----------------------------------------------------
def iter_items(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ----------------------------------------------------
# Main builder
# ----------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--out_index", default="my_hnsw.index")
    parser.add_argument("--out_meta", default="meta.jsonl")
    args = parser.parse_args()

    # Load encoder
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts, metas = [], []
    for i, item in enumerate(iter_items(args.input)):
        if i >= args.n:
            break

        t = make_text(item)
        if not t:
            continue

        texts.append(t)
        metas.append({
            "id": len(texts) - 1,
            "title": item.get("title"),
            "main_category": item.get("main_category"),
            "snippet": t[:200]
        })

    # Encode
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype("float32")

    # Normalize for cosine = L2
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    M = 32

    # Feder-compatible FAISS HNSW (MUST BE L2)
    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_L2)
    index.hnsw.efConstruction = 200

    # Add vectors
    index.add(embeddings)

    # Runtime search param
    index.hnsw.efSearch = 64

    # Save index
    faiss.write_index(index, args.out_index)

    # Save metadata
    with open(args.out_meta, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print("\nDone — saved HNSW index + metadata.")


if __name__ == "__main__":
    main()
