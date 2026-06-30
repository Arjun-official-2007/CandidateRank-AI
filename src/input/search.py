import chromadb

def candidate_rank(jd_text, top_k=100):

    client = chromadb.PersistentClient(path="./chromadb")
    collection = client.get_or_create_collection("candidates")

    Candidate_Rank=collection.query(
    query_texts=[jd_text],
    n_results=top_k)
    return Candidate_Rank
