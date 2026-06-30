import chromadb
from load_data import load_all_candidates

texts,ids,metadatas = load_all_candidates("data/candidates.jsonl")

client = chromadb.PersistentClient(path="./chromadb")

collection = client.get_or_create_collection("candidates")

BATCH_SIZE = 500

for i in range(0, len(texts), BATCH_SIZE):
    collection.upsert(
        documents=texts[i:i+BATCH_SIZE],
        ids=ids[i:i+BATCH_SIZE],
        metadatas=metadatas[i:i+BATCH_SIZE]
    )
    print(f"Upserted {min(i+BATCH_SIZE, len(texts))}/{len(texts)}")

print(f"Upserted {len(ids)} candidate into chromadb")