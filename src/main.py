"""
main.py

End-to-end orchestration for CandidateRank AI.

Usage:
    python main.py --jd path/to/job_description.docx
"""
from dotenv import load_dotenv
import argparse
import asyncio
import csv
from concurrent.futures import ThreadPoolExecutor

from src.input.search import candidate_rank
from src.output.candidate_store import load_candidate_store
from src.output.jd_parser import parse_jd, is_disqualified
from src.output.behavioral_score import skill_score, career_score, availability_score, trust_score
from src.output.score_fusion import fuse, rank

load_dotenv()
# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_PATH     = "./chromadb"
COLLECTION_NAME = "candidates"
TOP_K           = 100  # number of candidates to retrieve from Chroma
OUTPUT_PATH     = "output.csv"
MAX_WORKERS     = 10   # concurrent threads for LLM calls

# ── Chroma ────────────────────────────────────────────────────────────────────

def query_chroma(jd_text: str) -> list[tuple[str, float]]:
    """
    Query Chroma with JD text, return list of (candidate_id, semantic_score).
    Semantic score = 1 - normalized_distance, clamped to [0, 1].
    """
    results = candidate_rank(jd_text, top_k=TOP_K)

    ids       = results["ids"][0]
    distances = results["distances"][0]

    # Chroma returns L2 distances; normalize to 0-1 similarity
    max_dist = max(distances) if distances else 1.0
    scored   = [
        (cid, round(1.0 - (dist / max_dist), 4))
        for cid, dist in zip(ids, distances)
    ]
    return scored


# ── Per-candidate async scoring ───────────────────────────────────────────────

async def score_one(
    candidate: dict,
    jd_fields: dict,
    semantic: float,
    disqualifiers: list[str],
    executor: ThreadPoolExecutor,
) -> dict:
    """
    Score a single candidate asynchronously.
    LLM-heavy calls run in thread pool. Rule-based calls run inline.
    """
    loop = asyncio.get_event_loop()

    required_skills  = jd_fields["required_skills"]
    preferred_skills = jd_fields["preferred_skills"]
    min_exp          = jd_fields["min_exp"] or 0
    max_exp          = jd_fields["max_exp"] or 99
    work_type        = jd_fields["work_type"] or "onsite"
    job_location     = jd_fields["job_location"] or ""

    # LLM-heavy — run concurrently in threads
    skill_fut  = loop.run_in_executor(
        executor, skill_score, candidate, required_skills, preferred_skills
    )
    career_fut = loop.run_in_executor(
        executor, career_score, candidate, min_exp, max_exp
    )
    disq_fut   = loop.run_in_executor(
        executor, is_disqualified, candidate, disqualifiers
    )

    # Rule-based — pure math, run inline while LLM calls are in flight
    avail = availability_score(candidate, work_type, job_location)
    trust = trust_score(candidate)

    # Await all LLM results
    skill_result, career_result, (disqualified, _) = await asyncio.gather(
        skill_fut, career_fut, disq_fut
    )

    recruiter_response_rate = candidate["redrob_signals"]["recruiter_response_rate"]

    return fuse(
        candidate_id            = candidate["candidate_id"],
        candidate               = candidate,
        skill_result            = skill_result,
        career_result           = career_result,
        availability            = avail,
        trust                   = trust,
        semantic                = semantic,
        recruiter_response_rate = recruiter_response_rate,
        disqualified            = disqualified,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

async def run(jd_path: str):
    print(f"[1/6] Parsing JD: {jd_path}")
    jd_fields = parse_jd(jd_path)
    print(f"      Required skills : {jd_fields['required_skills']}")
    print(f"      Disqualifiers   : {jd_fields['disqualifiers']}")

    # Extract JD text for Chroma query
    from docx import Document
    doc     = Document(jd_path)
    jd_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    print(f"[2/6] Querying Chroma for top {TOP_K} candidates...")
    chroma_results = query_chroma(jd_text)
    candidate_ids  = [cid for cid, _ in chroma_results]
    semantic_map   = {cid: score for cid, score in chroma_results}
    print(f"      Retrieved {len(candidate_ids)} candidates")

    print("[3/6] Loading candidate store...")
    candidate_store = load_candidate_store("data/candidates.jsonl")
    print(f"      Store size: {len(candidate_store)} candidates")

    print("[4/6] Scoring candidates (async)...")
    disqualifiers = jd_fields.get("disqualifiers", [])

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        tasks = [
            score_one(
                candidate    = candidate_store[cid],
                jd_fields    = jd_fields,
                semantic     = semantic_map[cid],
                disqualifiers= disqualifiers,
                executor     = executor,
            )
            for cid in candidate_ids
            if cid in candidate_store
        ]
        fused_results = await asyncio.gather(*tasks)

    print(f"[5/6] Ranking {len(fused_results)} candidates...")
    ranked_rows = rank(list(fused_results))

    print(f"[6/6] Writing output to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(ranked_rows)

    print(f"\nDone. Top 5 candidates:")
    for row in ranked_rows[:5]:
        print(f"  #{row['rank']} {row['candidate_id']} — {row['score']} — {row['reasoning']}")


def main():
    parser = argparse.ArgumentParser(description="CandidateRank AI — Redrob Hackathon")
    parser.add_argument("--jd", required=True, help="Path to job_description.docx")
    args = parser.parse_args()

    asyncio.run(run(args.jd))


if __name__ == "__main__":
    main()