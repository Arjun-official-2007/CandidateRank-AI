"""
score_fusion.py

Receives pre-computed component scores per candidate, fuses them into a
final score, applies disqualifier cap, and returns a ranked output row.

Does NOT call any scoring functions — that's main.py's job.
"""

WEIGHTS = {
    "skill":        0.35,
    "career":       0.30,
    "availability": 0.20,
    "trust":        0.10,
    "semantic":     0.05,
}

DISQUALIFIER_CAP = 0.30  # hard ceiling if candidate is disqualified


def fuse(
    candidate_id: str,
    candidate: dict,
    skill_result: dict,
    career_result: dict,
    availability: float,
    trust: float,
    semantic: float,
    recruiter_response_rate: float,
    disqualified: bool = False,
) -> dict:
    """
    Fuse component scores into a final score for one candidate.

    Parameters
    ----------
    candidate_id            : str
    candidate               : raw candidate dict from candidate_store
    skill_result            : output of skill_score()  — expects keys: "score", "required_matched"
    career_result           : output of career_score() — expects keys: "score", "exp_score",
                              "trajectory_score", "role_align_score", "trajectory_reasoning",
                              "role_align_reasoning"
    availability            : float from availability_score()
    trust                   : float from trust_score()
    semantic                : float from Chroma (1 - normalized_distance), range 0–1
    recruiter_response_rate : float, raw field from candidate["redrob_signals"]
    disqualified            : bool, set by main.py after is_disqualified() check

    Returns
    -------
    dict with keys: candidate_id, final_score, component_scores, disqualified, reasoning
    """
    skill_score   = float(skill_result.get("score", 0.0))
    career_score  = float(career_result.get("score", 0.0))
    availability  = float(availability)
    trust         = float(trust)
    semantic      = float(semantic)

    raw_score = (
        WEIGHTS["skill"]        * skill_score  +
        WEIGHTS["career"]       * career_score +
        WEIGHTS["availability"] * availability +
        WEIGHTS["trust"]        * trust        +
        WEIGHTS["semantic"]     * semantic
    )

    final_score = min(raw_score, DISQUALIFIER_CAP) if disqualified else raw_score
    final_score = round(final_score, 4)

    reasoning = _build_reasoning(skill_result, career_result, candidate, recruiter_response_rate)

    return {
        "candidate_id": candidate_id,
        "final_score":  final_score,
        "component_scores": {
            "skill":        round(skill_score, 4),
            "career":       round(career_score, 4),
            "availability": round(availability, 4),
            "trust":        round(trust, 4),
            "semantic":     round(semantic, 4),
        },
        "disqualified": disqualified,
        "reasoning":    reasoning,
    }


def _build_reasoning(
    skill_result: dict,
    career_result: dict,
    candidate: dict,
    recruiter_response_rate: float,
) -> str:
    """
    Builds the one-liner reasoning string for the CSV output.
    Format: 'HR Manager with 6.1 yrs; 9 AI core skills; response rate 0.76.'
    """
    current_role   = candidate["profile"]["current_title"]
    years_exp      = candidate["profile"]["years_of_experience"]
    matched_skills = skill_result.get("required_matched", 0)
    response_rate  = round(recruiter_response_rate, 2)

    return (
        f"{current_role} with {years_exp:.1f} yrs; "
        f"{matched_skills} AI core skills; "
        f"response rate {response_rate:.2f}."
    )


def rank(fused_results: list[dict]) -> list[dict]:
    sorted_results = sorted(fused_results, key=lambda x: x["final_score"], reverse=True)

    output_rows = []
    for i, result in enumerate(sorted_results, start=1):
        output_rows.append({
            "candidate_id": result["candidate_id"],
            "rank":         i,
            "score":        result["final_score"],
            "reasoning":    result["reasoning"],
        })

    return output_rows