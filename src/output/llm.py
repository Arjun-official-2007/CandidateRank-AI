import os
import json
import re

# ── Switch provider here and nowhere else ──────────────────────────────────────
PROVIDER = "groq"          # "groq" | "anthropic"
GROQ_MODEL = "llama-3.3-70b-versatile"
ANTHROPIC_MODEL = "claude-sonnet-4-6"
# ──────────────────────────────────────────────────────────────────────────────

# Fallback returned when the LLM call fails or returns unparseable JSON
_ERROR_RESPONSE = {"score": None, "reasoning": "llm_error"}


def _strip_fences(text: str) -> str:
    text = text.strip()
    # Handles ```json ... ``` and ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def llm_call(prompt: str, system: str = None, temperature: float = 0.0) -> dict:
    """
    Send a prompt to the configured LLM provider and return parsed JSON.

    Every prompt in this project should ask the model to respond ONLY with a
    JSON object — no preamble, no markdown fences.  This wrapper strips fences
    defensively and returns _ERROR_RESPONSE on any failure so callers never
    crash the pipeline.

    Args:
        prompt      : The user-facing prompt text.
        system      : Optional system prompt (instructions / persona).
        temperature : 0.0 for deterministic scoring (default).

    Returns:
        Parsed dict from the model's JSON response, or _ERROR_RESPONSE on failure.
    """
    try:
        raw_text = _call_provider(prompt, system, temperature)
        clean    = _strip_fences(raw_text)
        return json.loads(clean)

    except json.JSONDecodeError:
        print(f"[llm.py] JSON parse error. Raw response:\n{raw_text}")
        return _ERROR_RESPONSE

    except Exception as e:
        print(f"[llm.py] LLM call failed: {e}")
        return _ERROR_RESPONSE


# ── Provider implementations ───────────────────────────────────────────────────

def _call_provider(prompt: str, system: str, temperature: float) -> str:
    """Route to the correct provider and return raw response text."""
    if PROVIDER == "groq":
        return _call_groq(prompt, system, temperature)
    elif PROVIDER == "anthropic":
        return _call_anthropic(prompt, system, temperature)
    else:
        raise ValueError(f"Unknown PROVIDER: {PROVIDER!r}")


def _call_groq(prompt: str, system: str, temperature: float) -> str:
    from groq import Groq

    client   = Groq(api_key=os.environ["GROQ_API_KEY"])
    messages = []

    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model       = GROQ_MODEL,
        messages    = messages,
        temperature = temperature,
    )
    return response.choices[0].message.content


def _call_anthropic(prompt: str, system: str, temperature: float) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    kwargs = dict(
        model       = ANTHROPIC_MODEL,
        max_tokens  = 1024,
        temperature = temperature,
        messages    = [{"role": "user", "content": prompt}],
    )
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    return response.content[0].text
