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
    """
    This function converts a raw JSON dictionary into a single searchable string.
    In vector search, we can't search 'fields' easily, so we squash them into one block of context.
    """
    # Define which fields from our raw data are useful for search
    fields = ["title", "subtitle", "description", "features", "details"]
    chunks = []
    
    for k in fields:
        v = item.get(k)
        if not v:
            continue
        # If the field is a list (like features), join them with a separator
        if isinstance(v, list):
            chunks.append(" ; ".join(str(x) for x in v))
        else:
            chunks.append(str(v))
            
    if not chunks:
        return None
        
    # Join everything with a pipe '|' to create a clean 'document' for the embedding model
    return " | ".join(chunks)


# ----------------------------------------------------
# Iterate through JSONL or JSONL.gz
# ----------------------------------------------------
def iter_items(path):
    """
    A generator that reads JSON Lines files. 
    Using gzip.open allows us to handle compressed data without decompressing it on disk first.
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                # Convert the text line back into a Python dictionary
                yield json.loads(line)


# ----------------------------------------------------
# Main builder
# ----------------------------------------------------
def main():
    # Setup command line arguments so we can change inputs/outputs without editing code
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)     # The path to your raw JSONL data
    parser.add_argument("--n", type=int, default=1000) # Limit the number of items to index (for testing)
    parser.add_argument("--out_index", default="my_hnsw.index") # Where to save the FAISS graph
    parser.add_argument("--out_meta", default="meta.jsonl")     # Where to save the text/ID mapping
    args = parser.parse_args()

    # Load the transformer model that turns text into math (vectors)
    # 'all-MiniLM-L6-v2' is small, fast, and great for general purpose use cases
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts, metas = [], []
    for i, item in enumerate(iter_items(args.input)):
        if i >= args.n:
            break

        # Process the raw item into a clean text string
        t = make_text(item)
        if not t:
            continue

        texts.append(t)
        # We store metadata separately because FAISS only stores the vectors (numbers)
        metas.append({
            "id": len(texts) - 1, # The 'row number' in the FAISS index
            "title": item.get("title"),
            "main_category": item.get("main_category"),
            "snippet": t[:200]    # A short preview for display purposes
        })

    # ENCODING: This turns our list of strings into a [N x 384] numpy matrix
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype("float32") # FAISS requires 32-bit floats for performance

    # NORMALIZATION: This is the 'Magic Trick'
    # By scaling every vector to a length of 1.0, we can use simple Euclidean (L2) 
    # distance to find things that are closest by Cosine Similarity (angle).
    faiss.normalize_L2(embeddings)

    # Define the dimensionality (384 for MiniLM)
    dim = embeddings.shape[1]
    
    # HNSW PARAMETERS:
    # M = The number of 'bridges' or links each data point has to its neighbors.
    # Higher M = higher accuracy, but larger file size on disk. 32 is a standard sweet spot.
    M = 32

    # Initialize the HNSW index using L2 metric (because vectors are normalized)
    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_L2)
    
    # efConstruction: How much 'effort' to put into building the graph connections.
    # Higher values create a more 'navigable' graph for the search algorithm.
    index.hnsw.efConstruction = 200

    # Actually build the graph by adding our vectors to it
    index.add(embeddings)

    # efSearch: How many neighbors to explore during a real search.
    # This is a runtime parameter that balances speed vs. accuracy.
    index.hnsw.efSearch = 64

    # Save the physical graph structure to a file
    faiss.write_index(index, args.out_index)

    # Save the metadata so when FAISS returns index '42', we know what text that was
    with open(args.out_meta, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print("\nDone — saved HNSW index + metadata.")


if __name__ == "__main__":
    main()