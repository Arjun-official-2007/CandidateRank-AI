"""
jd_parser.py

Two responsibilities:
1. parse_jd(path)          — extract structured fields from a .docx JD via LLM
2. is_disqualified(candidate, disqualifiers) — one LLM call per candidate to check
                                               if any disqualifier is tripped
"""

import json
from pathlib import Path
from docx import Document
from src.output.llm import llm_call

def _extract_text(docx_path: str) -> str:
    doc = Document(docx_path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

_PARSE_PROMPT = """
You are a structured data extractor. Given the job description below, extract the following fields exactly.

Return ONLY a valid JSON object with these keys:
- "min_exp": minimum years of experience required (float or null)
- "max_exp": maximum years of experience required (float or null)
- "work_type": one of "remote", "hybrid", "onsite", or null
- "job_location": city or region string, or null if fully remote
- "required_skills": list of skill name strings the candidate must have
- "preferred_skills": list of skill name strings that are nice-to-have
- "disqualifiers": list of plain-English disqualifier strings (conditions that
  immediately rule a candidate out, e.g. "entire career in consulting",
  "no product company experience"). Empty list if none stated.

Rules:
- If a field is not mentioned, use null or empty list as appropriate.
- Do not infer or hallucinate. Only extract what is explicitly or strongly implied.
- Skills must be atomic (e.g. "Python", not "Python programming experience").
- Disqualifiers must be concise conditions, not full sentences.

Job Description:
{jd_text}
"""


def parse_jd(docx_path: str) -> dict:
    """
    Extract structured fields from a job description .docx file.

    Returns
    -------
    dict with keys: min_exp, max_exp, work_type, job_location,
                    required_skills, preferred_skills, disqualifiers
    """
    jd_text = _extract_text(docx_path)
    prompt  = _PARSE_PROMPT.format(jd_text=jd_text)
    result  = llm_call(prompt)

    if result.get("score") is None:
        # llm_error fallback — return safe defaults
        return {
            "min_exp": None,
            "max_exp": None,
            "work_type": None,
            "job_location": None,
            "required_skills": [],
            "preferred_skills": [],
            "disqualifiers": [],
        }

    # call_llm returns {"score": <parsed dict or str>, "reasoning": ...}
    # For jd_parser we abuse the score field to carry the full parsed object.
    parsed = result["score"]
    if isinstance(parsed, str):
        parsed = json.loads(parsed)

    return parsed


#Disqualifier check
_DISQUALIFY_PROMPT = """
You are a strict recruiter screening assistant.

Given the candidate profile and a list of disqualifiers, determine if the candidate
is disqualified from consideration.

Disqualifiers:
{disqualifiers}

Candidate Profile:
{profile}

Instructions:
- A candidate is disqualified if ANY single disqualifier clearly applies to them.
- Be strict but fair. Only disqualify on clear evidence in the profile.
- Do not infer beyond what the profile states.

Return ONLY a valid JSON object:
{{
  "disqualified": true or false,
  "reason": "which disqualifier was tripped and why, or empty string if not disqualified"
}}
"""


def _build_candidate_payload(candidate: dict) -> dict:
    """
    Extract only the fields relevant for disqualifier evaluation.
    Keeps token cost low — ~200-400 tokens per candidate.
    """
    profile        = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])
    education      = candidate.get("education", [])
    signals        = candidate.get("redrob_signals", {})

    return {
        "current_title":       profile.get("current_title"),
        "years_of_experience": profile.get("years_of_experience"),
        "current_industry":    profile.get("current_industry"),
        "location":            profile.get("location"),
        "job_titles_history":  [c.get("title") for c in career_history],
        "industries_history":  [c.get("industry") for c in career_history],
        "skills":              [s.get("name") for s in candidate.get("skills", [])],
        "education":           [
            {
                "degree": e.get("degree"),
                "field":  e.get("field_of_study"),
                "tier":   e.get("tier"),
            }
            for e in education
        ],
        "preferred_work_mode": signals.get("preferred_work_mode"),
        "willing_to_relocate": signals.get("willing_to_relocate"),
    }


def is_disqualified(candidate: dict, disqualifiers: list[str]) -> tuple[bool, str]:
    """
    Check if a candidate trips any disqualifier via a single LLM call.

    Parameters
    ----------
    candidate      : raw candidate dict from candidate_store
    disqualifiers  : list of plain-English disqualifier strings from parse_jd()
    Returns
    -------
    (disqualified: bool, reason: str)
    """
    if not disqualifiers:
        return False, ""

    payload = _build_candidate_payload(candidate)
    prompt  = _DISQUALIFY_PROMPT.format(
        disqualifiers="\n".join(f"- {d}" for d in disqualifiers),
        profile=json.dumps(payload, indent=2),
    )

    result = llm_call(prompt)

    if result.get("score") is None:
        # On LLM error, default to not disqualified — don't penalise on uncertainty
        return False, "llm_error"

    parsed = result["score"]
    if isinstance(parsed, str):
        parsed = json.loads(parsed)

    return bool(parsed.get("disqualified", False)), parsed.get("reason", "")