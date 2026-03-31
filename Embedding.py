

from sentence_transformers import SentenceTransformer
import json
import faiss
import numpy as np

print("🔄 Loading embedding model...")
model = SentenceTransformer("BAAI/bge-m3")


print("🔄 Loading dataset...")
with open("data/rai_dataset_v1.json", "r", encoding="utf-8") as f:
    data = json.load(f)



def format_text(item):
    item_type = item.get("type", "").lower()

   
    if item_type == "QA":
        question = item.get("question", "")
        answer = item.get("answer", "") or item.get("content", "")

        if question and answer:
            return f"Question: {question} Answer: {answer}"
        else:
            return None

    # ✅ CONTENT
    elif item_type == "content":
        topic = item.get("topic", "General")
        content = item.get("content", "")

        if content:
            return f"{topic}: {content}"
        else:
            return None

    # ❌ skip data แปลกๆ
    return None


# 
print("🔄 Formatting text...")
texts = []

for item in data:
    text = format_text(item)
    if text:
        texts.append(text)

print(f"✅ Total usable texts: {len(texts)}")


print("🔄 Creating embeddings...")
embeddings = model.encode(texts, normalize_embeddings=True)


print("🔄 Building FAISS index...")
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)  # cosine similarity
index.add(np.array(embeddings))



print("💾 Saving files...")

# vector DB
faiss.write_index(index, "rai_index.faiss")

# text mapping
with open("texts.json", "w", encoding="utf-8") as f:
    json.dump(texts, f, ensure_ascii=False, indent=2)

# raw data (optional แต่ดีมาก)
with open("raw_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)



print("🎉 DONE! Everything is ready for RAG")
print("📦 Files created:")
print("- rai_index.faiss")
print("- texts.json")
print("- raw_data.json")