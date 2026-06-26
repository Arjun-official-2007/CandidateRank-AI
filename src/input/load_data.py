import json

def Candidate_stream(file_path):
    required = {"candidate_id", "profile", "career_history", "education", "skills"}
    with open(file_path,'r',encoding="utf-8-sig") as file:
        for line_number,line in enumerate(file,1):
            clean_line = line.strip()

            if not clean_line:
                continue

            try:
                profile=json.loads(clean_line)
            except json.JSONDecodeError:
                print(f"Skipping malformed JSON on line {line_number}")
                continue

            if not required.issubset(profile.keys()):
                missing = required - profile.keys()
                print(f"Skiping Line:{line_number}: Missing Field:{missing}")
                continue
                
            yield profile
        

def build_candidate_record(candidate):
    text= candidate["profile"]["headline"] + " " + \
            candidate["profile"]["summary"] + " " + \
            candidate["profile"]["current_title"] + " " + \
            candidate["profile"]["current_industry"] + " " + \
            " ".join([role["title"] + " " + role["description"] for role in candidate["career_history"]]) + " " + \
            " ".join([skill["name"] for skill in candidate["skills"]]) + " " + \
            " ".join([edu["degree"] + " " + edu["field_of_study"] for edu in candidate["education"]])
    id=candidate["candidate_id"]
    metadata={
        "name": candidate["profile"]["anonymized_name"],
        "current_title": candidate["profile"]["current_title"],
        "current_company": candidate["profile"]["current_company"],
        "location": candidate["profile"]["location"],
        "years_of_experience": candidate["profile"]["years_of_experience"]
    }
    return text,id,metadata


def load_all_candidates(file_path):

    texts,ids,metadatas=[],[],[]
    for candidate in Candidate_stream(file_path):
        text,id,metadata=build_candidate_record(candidate)
        texts.append(text)
        ids.append(id)
        metadatas.append(metadata)
    return texts,ids,metadatas

if __name__ == "__main__":
    texts, ids, metadatas = load_all_candidates(r"..\..\data\candidates.jsonl")
    print(f"Loaded {len(texts)} candidates")
    print(texts[0])
    print(ids[0])
    print(metadatas[0])