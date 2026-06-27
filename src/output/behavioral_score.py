import json
from datetime import datetime
from llm import llm_call

PROFICIENCY_RANK = {"beginner": 0, "intermediate": 1, "advanced": 2}
 
# ── System prompt shared across all scoring calls ─────────────────────────────
SCORER_SYSTEM = (
    "You are a structured data extractor for a recruitment scoring system. "
    "Respond ONLY with a valid JSON object. No explanation, no markdown fences, no preamble."
)

def trust_score(candidate):
    verified_email = candidate["redrob_signals"]["verified_email"]
    verified_phone = candidate["redrob_signals"]["verified_phone"]
    interview_completion_rate = candidate["redrob_signals"]["interview_completion_rate"]
    recruiter_response_rate = candidate["redrob_signals"]["recruiter_response_rate"]

    email_score = 1 if verified_email else 0
    phone_score = 1 if verified_phone else 0
    
    final_score = (email_score * 0.15) + (phone_score * 0.15) + (interview_completion_rate * 0.4) + (recruiter_response_rate * 0.3)

    return final_score

def availability_score(candidate, work_type, job_location):
    signals = candidate["redrob_signals"]
    profile = candidate["profile"]

    notice_period = min(signals["notice_period_days"], 90)
    notice_score = 1 - (notice_period/ 90) 

    open_to_work = 1 if signals["open_to_work_flag"] else 0

    last_active = datetime.strptime(
        signals["last_active_date"],
        "%Y-%m-%d"
    )

    days_since_last_active = min((datetime.now() - last_active).days, 180)

    active_score = 1 - (days_since_last_active / 180)

    if work_type == "remote":
        return (open_to_work * 0.35) + (notice_score * 0.35) + (active_score * 0.3)
    
    candidate_location = profile["location"]   
    location_match = 1 if job_location == candidate_location else 0

    if not location_match:
        location_score = 1 if signals["willing_to_relocate"] else 0
    else:
        location_score = 1

    if work_type == "hybrid":
        return (open_to_work * 0.325) + (notice_score * 0.325) + (active_score * 0.25) + (location_score * 0.1)
    
    return (open_to_work * 0.25) + (notice_score * 0.25) + (active_score * 0.2) + (location_score * 0.3)


def career_score(candidate, min_exp, max_exp):
    # ── 1. Years of experience (rule-based) ───────────────────────────────────
    years = candidate["profile"]["years_of_experience"]
 
    if min_exp <= years <= max_exp:
        exp_score = 1.0
    else:
        distance  = min(abs(years - min_exp), abs(years - max_exp))
        exp_score = max(0.0, 1.0 - (distance / max_exp))
 
    # ── 2. Career trajectory (LLM) ────────────────────────────────────────────
    career_history = [
        {
            "title"        : role["title"],
            "company"      : role["company"],
            "company_size" : role["company_size"],
            "duration_months": role["duration_months"],
            "is_current"   : role["is_current"],
        }
        for role in candidate["career_history"]
    ]
 
    trajectory_prompt = f"""
Evaluate the career growth trajectory of this candidate based on their work history.
Consider: title progression (seniority increases), company quality/size growth, and tenure stability.
A candidate who stays at the same title for many years with no progression is stagnant, even at a top company.
 
Career history (chronological):
{json.dumps(career_history, indent=2)}
 
Return ONLY this JSON:
{{
  "score": <float 0.0 to 1.0>,
  "reasoning": "<one sentence justification>"
}}
 
Score guide:
1.0 = clear upward progression in both title and company quality
0.7 = moderate progression, some stagnation
0.4 = mostly flat, minimal growth signals
0.1 = no progression or declining trajectory
"""
    trajectory_result  = llm_call(trajectory_prompt, system=SCORER_SYSTEM)
    trajectory_score   = float(trajectory_result.get("score") or 0.5)
    trajectory_reason  = trajectory_result.get("reasoning", "llm_error")
 
    # ── 3. Current role alignment (LLM — category-based) ─────────────────────
    # Fixed category → score mapping keeps LLM output consistent across candidates
    ALIGN_SCORE_MAP = {
        "exact"     : 1.0,
        "adjacent"  : 0.8,   # same domain, different seniority
        "related"   : 0.6,   # transferable domain
        "loose"     : 0.3,
        "unrelated" : 0.0,
    }
 
    role_prompt = f"""
Compare the candidate's current role to the job description role and classify the alignment.
 
Candidate current title    : {candidate["profile"]["current_title"]}
Candidate current industry : {candidate["profile"]["current_industry"]}
Job title being hired for  : Senior AI Engineer
Job industry               : AI / HR-Tech / SaaS
 
Pick EXACTLY one category:
- "exact"      — same or near-identical role and domain
- "adjacent"   — same domain but different seniority level (over or under)
- "related"    — different title but clearly transferable skills and domain
- "loose"      — loosely related, requires significant pivot
- "unrelated"  — no meaningful overlap
 
Return ONLY this JSON:
{{
  "category": "<one of the five categories above>",
  "reasoning": "<one sentence justification>"
}}
"""
    role_result      = llm_call(role_prompt, system=SCORER_SYSTEM)
    category         = role_result.get("category", "loose")
    role_align_score = ALIGN_SCORE_MAP.get(category, 0.3)
    role_align_reason = role_result.get("reasoning", "llm_error")
 
    # ── 4. Weighted aggregation ───────────────────────────────────────────────
    final_score = (
        0.40 * exp_score        +
        0.35 * trajectory_score +
        0.25 * role_align_score
    )
 
    return {
        "score"                : round(final_score, 4),
        "exp_score"            : round(exp_score, 4),
        "trajectory_score"     : round(trajectory_score, 4),
        "role_align_score"     : round(role_align_score, 4),
        "trajectory_reasoning" : trajectory_reason,
        "role_align_reasoning" : role_align_reason,
    }

def skill_score(candidate, required_skills: list[dict], preferred_skills: list[dict]):
    """
    Single LLM call that:
      1. Semantically matches candidate skills to JD required + preferred skills
      2. Evaluates proficiency (if JD specifies it) using proficiency ladder
      3. Evaluates duration (if JD specifies it) using distance penalty
 
    Proficiency and duration act as multipliers on the required match ratio,
    not additive components — preserving the 0.8 / 0.2 weight split.
 
    Args:
        candidate        : full candidate dict
        required_skills  : list of {name, proficiency, min_duration_months}
        preferred_skills : list of {name, proficiency, min_duration_months}
 
    Returns:
        {
            "score"                : float 0-1,
            "required_match_ratio" : float,
            "preferred_match_ratio": float,
            "proficiency_score"    : float | None,
            "duration_score"       : float | None,
            "reasoning"            : str
        }
    """
 
    candidate_skills = candidate["skills"]  # list of {name, proficiency, endorsements, duration_months}
 
    # Determine what optional components the JD actually asks for
    jd_asks_proficiency = any(s.get("proficiency") for s in required_skills)
    jd_asks_duration    = any(s.get("min_duration_months") for s in required_skills)
 
    # ── Build the LLM prompt ──────────────────────────────────────────────────
    optional_instructions = ""
    optional_output_fields = ""
 
    if jd_asks_proficiency:
        optional_instructions += """
- For each matched required skill where the JD specifies proficiency:
  Candidate meets or exceeds → full credit. One level below → 0.5 credit. Two levels below → 0 credit.
  Proficiency ladder: beginner < intermediate < advanced.
  Aggregate into a single proficiency_score (0.0 to 1.0) across all required skills with proficiency specified.
"""
        optional_output_fields += '  "proficiency_score": <float 0.0 to 1.0>,\n'
 
    if jd_asks_duration:
        optional_instructions += """
- For each matched required skill where the JD specifies min_duration_months:
  Use distance penalty: max(0, 1 - (shortfall / min_duration_months)).
  Shortfall = max(0, min_duration_months - candidate_duration_months).
  Aggregate into a single duration_score (0.0 to 1.0) across all required skills with duration specified.
"""
        optional_output_fields += '  "duration_score": <float 0.0 to 1.0>,\n'
 
    # Assessment scores (from redrob_signals) — fed as context if available
    assessment_scores = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    assessment_context = ""
    if assessment_scores:
        assessment_context = f"\nCandidate skill assessment scores (platform-verified): {json.dumps(assessment_scores)}\nUse these to validate or adjust proficiency judgement where skill names overlap."
 
    prompt = f"""
You are evaluating a candidate's skills against a job description.
 
CANDIDATE SKILLS:
{json.dumps(candidate_skills, indent=2)}
{assessment_context}
 
JD REQUIRED SKILLS:
{json.dumps(required_skills, indent=2)}
 
JD PREFERRED SKILLS:
{json.dumps(preferred_skills, indent=2)}
 
INSTRUCTIONS:
- Semantically match candidate skills to JD skills. "vector databases" matches "Chroma", "FAISS", "Pinecone" etc.
- Count how many required skills the candidate matches (required_matched / required_total).
- Count how many preferred skills the candidate matches (preferred_matched / preferred_total).
{optional_instructions}
- Write one sentence of reasoning summarising the skill fit.
 
Return ONLY this JSON:
{{
  "required_matched": <int>,
  "required_total": <int>,
  "preferred_matched": <int>,
  "preferred_total": <int>,
{optional_output_fields}  "reasoning": "<one sentence>"
}}
"""
 
    result = llm_call(prompt, system=SCORER_SYSTEM)
 
    # ── Parse LLM output ──────────────────────────────────────────────────────
    required_matched = int(result.get("required_matched") or 0)
    required_total   = int(result.get("required_total")   or max(len(required_skills), 1))
    preferred_matched= int(result.get("preferred_matched")or 0)
    preferred_total  = int(result.get("preferred_total")  or max(len(preferred_skills), 1))
 
    required_ratio   = required_matched / required_total   if required_total   else 0.0
    preferred_ratio  = preferred_matched / preferred_total if preferred_total  else 0.0
 
    proficiency_score = result.get("proficiency_score")   # None if JD didn't ask
    duration_score    = result.get("duration_score")      # None if JD didn't ask
 
    # ── Apply proficiency + duration as multipliers on required ratio ─────────
    # Each multiplier defined → dampens the required component if candidate falls short
    multiplier = 1.0
    active_multipliers = 0
 
    if proficiency_score is not None:
        multiplier += float(proficiency_score)
        active_multipliers += 1
    if duration_score is not None:
        multiplier += float(duration_score)
        active_multipliers += 1
 
    if active_multipliers > 0:
        multiplier = multiplier / (1 + active_multipliers)  # normalise back to 0-1 range
        required_component = required_ratio * multiplier
    else:
        required_component = required_ratio
 
    # ── Final score: 0.8 required + 0.2 preferred (capped bonus) ─────────────
    preferred_bonus = min(preferred_ratio * 0.2, 0.2)
    final_score     = min(1.0, (required_component * 0.8) + preferred_bonus)
 
    return {
        "score"                : round(final_score, 4),
        "required_match_ratio" : round(required_ratio, 4),
        "preferred_match_ratio": round(preferred_ratio, 4),
        "proficiency_score"    : round(float(proficiency_score), 4) if proficiency_score is not None else None,
        "duration_score"       : round(float(duration_score), 4)    if duration_score    is not None else None,
        "reasoning"            : result.get("reasoning", "llm_error"),
    }
    