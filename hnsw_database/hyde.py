from google import genai
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))

query = input("Search: ")

response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents="Generate a very short, neutral product description for embedding-based search." \
                        "Rules:" \
                        "- 1–2 sentences only" \
                        "- No marketing language" \
                        "- No adjectives unless they describe physical properties" \
                        "- Focus on brand, color, material, and function" \
                        "- Write in the style of a product catalog" \
                        ":"  + query
)

# Extract the text from the response
hyde_text = response.text
print("Generated description:", hyde_text)

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("my_hnsw.index")
index.hnsw.efSearch = 64  

meta = []
with open("meta.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        meta.append(json.loads(line))

query_vec = model.encode(
    [hyde_text],
    convert_to_numpy=True
).astype("float32")

faiss.normalize_L2(query_vec)

k = 10
distances, indices = index.search(query_vec, k)

for rank, idx in enumerate(indices[0]):
    m = meta[idx]
    print(
        f"{rank+1:02d}. "
        f"title={m['title']} | "
        f"category={m['main_category']} | "
        f"distance={distances[0][rank]:.4f}"
    )
