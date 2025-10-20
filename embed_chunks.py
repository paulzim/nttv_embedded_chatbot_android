import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import json
import os

# =========================================
# 1. Paths and setup
# =========================================

# Folder where your .txt files are stored
data_dir = Path(r"C:\Users\paulz\GitHub\nttv_embedded_chatbot_android\data")

# Folder where we'll save vectors.bin and meta.jsonl
output_dir = Path(r"C:\Users\paulz\GitHub\nttv_embedded_chatbot_android\android_assets")
output_dir.mkdir(parents=True, exist_ok=True)

# ✅ Open access embedding model — no token needed
MODEL_NAME = "intfloat/e5-small-v2"

# Load tokenizer + model
print(f"Loading model {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()
print("Model loaded ✅")

# =========================================
# 2. Load + chunk content
# =========================================

def chunk_text(text: str, max_chars=500, overlap=50):
    step = max_chars - overlap
    return [text[i:i+max_chars] for i in range(0, len(text), step)]

all_chunks = []
metadata = []

for file in data_dir.glob("*.txt"):
    # Handle encoding issues gracefully
    try:
        text = file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = file.read_text(encoding="cp1252")

    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        metadata.append({
            "file": file.name,
            "chunk_index": i,
            "text": chunk
        })

print(f"✅ Loaded {len(metadata)} chunks from {len(list(data_dir.glob('*.txt')))} files")

# =========================================
# 3. Embed chunks
# =========================================

embeddings = []
with torch.no_grad():
    for chunk in all_chunks:
        # The E5 family expects the query to start with this prefix for good results
        # (for documents, you can use "passage: " as well)
        prefixed = "passage: " + chunk

        inputs = tokenizer(prefixed, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        # L2 normalize for cosine similarity
        emb = emb / np.linalg.norm(emb)
        embeddings.append(emb.astype(np.float32))

embeddings = np.stack(embeddings)
print(f"✅ Generated embeddings with shape {embeddings.shape}")

# =========================================
# 4. Save vectors.bin (FP16) and meta.jsonl
# =========================================

# Convert to FP16 to save space
embeddings_fp16 = embeddings.astype(np.float16)
vectors_path = output_dir / "vectors.bin"
embeddings_fp16.tofile(vectors_path)

meta_path = output_dir / "meta.jsonl"
with meta_path.open("w", encoding="utf-8") as f:
    for i, meta in enumerate(metadata):
        meta_record = {
            "id": i,
            "file": meta["file"],
            "chunk_index": meta["chunk_index"],
            "dim": embeddings.shape[1],  # should be 384 for e5-small-v2
            "text": meta["text"]
        }
        f.write(json.dumps(meta_record, ensure_ascii=False) + "\n")

print(f"✅ Saved vectors to: {vectors_path}")
print(f"✅ Saved metadata to: {meta_path}")
