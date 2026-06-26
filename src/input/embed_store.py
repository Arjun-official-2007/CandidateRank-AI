import chromadb
from load_data import load_all_candidates

texts,ids,metadatas = load_all_candidates(r"..\..\data\candidates.jsonl")

client = chromadb.PersistentClient(path="./chromadb")

collection = client.get_or_create_collection("candidates")

collection.upsert(
    documents=texts,
    ids=ids,
    metadatas=metadatas
)

print(f"Upserted {len(ids)} candidate into chromadb")