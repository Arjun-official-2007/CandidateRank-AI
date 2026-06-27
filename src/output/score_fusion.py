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
    skill_result: dict,
    career_result: dict,
    availability: float,
    trust: float,
    recruiter_response_rate: float,
    semantic: float,
    disqualified: bool = False,
) -> dict:
    """
    Fuse component scores into a final score for one candidate.
    """

    skill_score    = float(skill_result.get("score", 0.0))
    career_score   = float(career_result.get("final_score", 0.0))
    availability   = float(availability)
    trust          = float(trust)
    semantic       = float(semantic)

    raw_score = (
        WEIGHTS["skill"]        * skill_score  +
        WEIGHTS["career"]       * career_score +
        WEIGHTS["availability"] * availability +
        WEIGHTS["trust"]        * trust        +
        WEIGHTS["semantic"]     * semantic
    )

    final_score = min(raw_score, DISQUALIFIER_CAP) if disqualified else raw_score
    final_score = round(final_score, 4)

    reasoning = _build_reasoning(skill_result, career_result, recruiter_response_rate)

    return {
        "candidate_id": candidate_id,
        "final_score": final_score,
        "component_scores": {
            "skill":        round(skill_score, 4),
            "career":       round(career_score, 4),
            "availability": round(availability, 4),
            "trust":        round(trust, 4),
            "semantic":     round(semantic, 4),
        },
        "disqualified": disqualified,
        "reasoning": reasoning,
    }


def _build_reasoning(skill_result: dict, career_result: dict, recruiter_response_rate: float) -> str:
    """
    Builds the one-liner reasoning string for the CSV output.
    """
    # Current role — from role_alignment block inside career_result
    role_alignment = career_result.get("component_scores", {}).get("role_alignment", {})
    current_role   = role_alignment.get("current_role", "Unknown Role")

    # Years experience
    years_exp = career_result.get("years_experience", 0.0)

    # Matched required skill count — skill_result["candidate_skills"] is the raw list
    matched_skills = skill_result.get("matched_required_count", 0)

    return (
        f"{current_role} with {years_exp:.1f} yrs; "
        f"{matched_skills} AI core skills; "
        f"response rate {recruiter_response_rate:.2f}."
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