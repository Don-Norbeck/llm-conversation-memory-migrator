"""
core/summarizer.py
Uses Ollama (local) to summarize individual conversations.
100% local — requires Ollama running on this machine.
"""

import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional


# ── Ollama local summarization ───────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2"

SUMMARIZE_PROMPT_CONCISE = """You are analyzing a conversation between a user and an AI assistant.

Extract and return ONLY a JSON object with these fields:
{{
  "topics": ["list", "of", "main", "topics"],
  "summary": "4-6 sentence summary. Include: specific names of people, companies, tools, or frameworks mentioned; specific decisions made and conclusions reached; specific file names, project names, or artifacts created; and any open questions or unresolved threads.",
  "key_decisions": ["important decisions or conclusions reached"],
  "artifacts": ["code, documents, frameworks, or other outputs created — use specific names"],
  "open_threads": ["unresolved questions or next steps mentioned"],
  "preferences": ["user preferences, style notes, or behavioral patterns observed"],
  "bucket": "single best category from the list below"
}}

Bucket definitions — pick the single best match:
- Work & Career: jobs, resumes, interviews, career planning, professional networking, LinkedIn
- Creative Projects: writing, design, art, music, games, creative tools, storytelling
- Technical & Coding: code, programming, scripts, software, hardware, AI projects, local AI setup
- Research & Learning: articles, research, education, learning new topics, summaries
- Health & Wellness: medical, fitness, diet, mental health, diabetes, medications
- Personal & Family: family, relationships, personal life, home, civic involvement
- Finance & Legal: money, taxes, legal questions, budgeting, contracts
- Hobbies & Interests: sports, collecting, skiing, vinyl records, sneakers, cards, cooking, games
- Travel & Planning: trips, destinations, travel logistics, hotels, flights
- General: anything that does not clearly fit the above categories

Return ONLY valid JSON. No preamble. No explanation.

Conversation title: {title}

Conversation:
{text}
"""

SUMMARIZE_PROMPT_DETAILED = """You are analyzing a conversation between a user and an AI assistant.

Extract and return ONLY a JSON object with these fields:
{{
  "topics": ["list", "of", "main", "topics"],
  "summary": "Write 8-20 bullet points (as a single string, each bullet on its own line starting with '• '). Cover: specific names of people, companies, tools, and frameworks mentioned; every specific decision made and conclusion reached; every specific file name, project name, or artifact created; all open questions and unresolved threads; and any notable context or background established.",
  "key_decisions": ["important decisions or conclusions reached — be specific"],
  "artifacts": ["code, documents, frameworks, or other outputs created — use specific names"],
  "open_threads": ["unresolved questions or next steps mentioned"],
  "preferences": ["user preferences, style notes, or behavioral patterns observed"],
  "bucket": "single best category from the list below"
}}

Bucket definitions — pick the single best match:
- Work & Career: jobs, resumes, interviews, career planning, professional networking, LinkedIn
- Creative Projects: writing, design, art, music, games, creative tools, storytelling
- Technical & Coding: code, programming, scripts, software, hardware, AI projects, local AI setup
- Research & Learning: articles, research, education, learning new topics, summaries
- Health & Wellness: medical, fitness, diet, mental health, diabetes, medications
- Personal & Family: family, relationships, personal life, home, civic involvement
- Finance & Legal: money, taxes, legal questions, budgeting, contracts
- Hobbies & Interests: sports, collecting, skiing, vinyl records, sneakers, cards, cooking, games
- Travel & Planning: trips, destinations, travel logistics, hotels, flights
- General: anything that does not clearly fit the above categories

Return ONLY valid JSON. No preamble. No explanation.

Conversation title: {title}

Conversation:
{text}
"""

# Default prompt alias
SUMMARIZE_PROMPT = SUMMARIZE_PROMPT_CONCISE


def _call_ollama(prompt: str, model: str = DEFAULT_MODEL) -> Optional[str]:
    """Make a request to the local Ollama API."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("response", "")
    except urllib.error.URLError as e:
        print(f"Ollama connection error: {e}")
        print("Is Ollama running? Start it with: ollama serve")
        return None
    except Exception as e:
        print(f"Ollama error: {e}")
        return None


def _truncate_conversation(convo: Dict[str, Any], max_chars: int = 6000) -> str:
    """
    Convert conversation messages to text, truncated to max_chars.
    Keeps first and last messages to preserve context bookends.
    """
    messages = convo.get("messages", [])
    if not messages:
        return ""

    lines = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['text'][:500]}")

    full_text = "\n".join(lines)
    if len(full_text) <= max_chars:
        return full_text

    # Truncate middle, keep start and end
    half = max_chars // 2
    return full_text[:half] + "\n\n[... truncated ...]\n\n" + full_text[-half:]


def _repair_json(raw: str) -> str:
    """
    Attempt to repair truncated JSON responses from Llama.
    Handles the most common failure mode: response cut off mid-string.
    """
    # If it doesn't start with { something is very wrong
    if not raw.startswith("{"):
        idx = raw.find("{")
        if idx == -1:
            return raw
        raw = raw[idx:]

    # Close any open strings by tracking unescaped structural quotes.
    # A simple toggle breaks when LLM outputs unescaped inner quotes inside
    # string values (e.g. `"the "pre-AI internet" era"`).  Instead, when we
    # are already inside a string and hit a `"`, peek at the next non-whitespace
    # character: if it looks like a JSON delimiter (`,`, `}`, `]`, `:`, `"`) or
    # we are at end-of-input, treat it as a real string-closer; otherwise treat
    # it as an unescaped content quote and stay inside the string.
    in_string = False
    escaped = False
    i = 0
    n = len(raw)
    while i < n:
        char = raw[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if char == '\\':
            escaped = True
            i += 1
            continue
        if char == '"':
            if not in_string:
                in_string = True
            else:
                # Look past whitespace to find the next structural character.
                j = i + 1
                while j < n and raw[j] in ' \t\r\n':
                    j += 1
                next_char = raw[j] if j < n else ''
                if next_char in ',}]:"' or j >= n:
                    in_string = False  # proper string closer
                # else: unescaped inner quote — remain in_string
        i += 1

    if in_string:
        raw += '"'

    # Close open arrays and objects
    open_brackets = raw.count('[') - raw.count(']')
    open_braces = raw.count('{') - raw.count('}')

    raw += ']' * max(0, open_brackets)
    raw += '}' * max(0, open_braces)

    return raw


def summarize_conversation(
    convo: Dict[str, Any],
    model: str = DEFAULT_MODEL,
    summary_style: str = "concise",
) -> Optional[Dict[str, Any]]:
    """
    Summarize a single normalized conversation using local Ollama.
    Returns a dict with summary fields, or None if summarization failed.
    """
    title = convo.get("title", "Untitled")
    text = _truncate_conversation(convo)

    if not text.strip():
        return None

    prompt_template = (
        SUMMARIZE_PROMPT_DETAILED if summary_style == "detailed" else SUMMARIZE_PROMPT_CONCISE
    )
    prompt = prompt_template.format(title=title, text=text)
    raw = _call_ollama(prompt, model=model)

    if not raw:
        return None

    # Parse JSON response
    try:
        # Strip markdown code fences if model wrapped response
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        clean = clean.strip()

        # Attempt to repair truncated JSON
        clean = _repair_json(clean)

        result = json.loads(clean)
        result["title"] = title
        result["conversation_id"] = convo.get("id", "")
        result["created"] = convo.get("created", "")
        result["updated"] = convo.get("updated", "")
        return result
    except json.JSONDecodeError as e:
        print(f"Failed to parse summary JSON for '{title}': {e}")
        print(f"Raw response: {raw[:200]}")
        return None


def summarize_all(
    conversations: list,
    model: str = DEFAULT_MODEL,
    progress_callback=None,
    summary_style: str = "concise",
) -> list:
    """
    Summarize a list of conversations using local Ollama.

    Args:
        conversations: List of normalized conversation dicts
        model: Ollama model name
        progress_callback: Optional function(current, total) for progress updates
        summary_style: "concise" (4-6 sentences) or "detailed" (8-20 bullets)

    Returns:
        List of summary dicts (failed summaries are excluded)
    """
    summaries = []
    total = len(conversations)

    for i, convo in enumerate(conversations):
        if progress_callback:
            progress_callback(i + 1, total)
        else:
            if i % 10 == 0:
                print(f"  Summarizing {i + 1}/{total} ...")

        summary = summarize_conversation(convo, model=model, summary_style=summary_style)
        if summary:
            summaries.append(summary)

    print(f"Successfully summarized {len(summaries)}/{total} conversations.")
    return summaries


def summarize_all_gen(
    conversations: list,
    model: str = DEFAULT_MODEL,
    summary_style: str = "concise",
):
    """
    Generator version of summarize_all.
    Yields (current, total, title, summary_or_None) after each conversation,
    allowing callers to stream progress updates in real time.
    """
    total = len(conversations)
    for i, convo in enumerate(conversations):
        title = convo.get("title", "Untitled")
        summary = summarize_conversation(convo, model=model, summary_style=summary_style)
        yield (i + 1, total, title, summary)


def check_ollama_running() -> bool:
    """Check if Ollama is running and accessible on localhost."""
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/tags",
            method="GET"
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False