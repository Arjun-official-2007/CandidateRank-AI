from src.input.load_data import Candidate_stream

def load_candidate_store(file_path):
    candidates = {}
    for candidate in Candidate_stream("data/candidates.jsonl"):
        candidates[candidate["candidate_id"]] = candidate
