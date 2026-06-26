from docx import Document
import chromadb

doc = Document(r"..\..\data\job_description.docx")
jd_text="\n".join([para.text for para in doc.paragraphs])

client = chromadb.PersistentClient(path="./chromadb")
collection = client.get_or_create_collection("candidates")

Candidate_Rank=collection.query(
    query_texts=[jd_text],
    n_results=20)
print(Candidate_Rank)
