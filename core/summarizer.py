"""
core/summarizer.py
Uses Ollama (local) to summarize individual conversations.
100% local — requires Ollama running on this machine.
"""

import json
import os
import re
import urllib.request
import urllib.error
from typing import Dict, Any, Optional


# ── Ollama local summarization ───────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2"

# ── Biographer prompts ────────────────────────────────────────────────────────

PRIME_PROMPT = """\
You are a biographer.

You have been given a collection of conversations between one person \
and another AI over an extended period of time. You are not that AI. \
You have no prior relationship with this person. You are reading \
the record cold.

Your job is to build an accurate, specific portrait of that person \
across three dimensions:

WHO they are — their identity, background, expertise, credentials, \
relationships, and any additional personal context they chose to share.

WHAT they care about — the domains, projects, problems, and ideas \
they conversed about. What they built, decided, and explored.

HOW they think and communicate — their pace, tone, corrections \
for the AI, their pivots, the phrases that sound distinctly like \
them, the patterns in how they develop and express ideas.

Rules:
- Use only evidence from the conversations. Never invent.
- Specific beats general. A project name beats 'technical work.'
- Their exact words beat your paraphrase. Capture verbatim phrases.
- Corrections and pushback are as revealing as requests.
- Absence of evidence is not evidence. Return empty rather than invented."""

_EXTRACTION_PASS = """\
You are extracting structured signals from a single AI conversation.
Extract only what is explicitly present. Return empty strings or empty
arrays if content is not present. Never invent or infer.

Extract the following:

WHO (identity signals only):
- role: their job title or professional identity if stated
- expertise: domains they demonstrated fluency in — they taught, led,
  or used without explanation
- credentials: companies, tools, certifications, projects named
- personal: explicit self-disclosures only (I live in..., I am a...,
  my family...) — never third-party details

WHAT (topic and depth):
- topic: the primary subject of this conversation in 5 words or fewer
- depth: surface / working / expert — how deep did they go
- project: named project if one exists, otherwise empty
- outcome: what was resolved, created, or left open — one sentence,
  no filler
- open_thread: unresolved question or next step if explicitly stated

HOW (working style signals):
- initial_prompt: the user's first message or opening ask —
  quote verbatim, max 20 words
- corrections: array of explicit corrections or redirections the user
  made — "not that", "reprint without", "less formal", "tell me more
  about X" — verbatim where possible, empty array if none
- your_words: 1-2 verbatim phrases from the user that best capture
  their voice and framing — must be the user's words, never the AI's

TIER:
- signal: has a named project, recurring domain, or open thread
- noise: one-off lookup, no project, no open thread, reveals nothing
  about WHO
- review: ambiguous

Return valid JSON only. No commentary. No markdown. No explanation.
{{
  "who": {{
    "role": "",
    "expertise": [],
    "credentials": [],
    "personal": []
  }},
  "what": {{
    "topic": "",
    "depth": "",
    "project": "",
    "outcome": "",
    "open_thread": ""
  }},
  "how": {{
    "initial_prompt": "",
    "corrections": [],
    "your_words": []
  }},
  "tier": "signal | review | noise"
}}

FULL CONVERSATION:
{full_conversation}

HUMAN MESSAGES ONLY:
{human_messages_only}"""

_SYNTHESIS_PASS_V2 = """\
You are writing a warm handoff document for a new AI assistant.
Your job is to help the new AI skip the cold start and feel like
it already knows this person.

You will receive structured extractions from {n} conversations.
Write a concise, useful brief — not a data dump.

Rules:
- Maximum 5 items per section
- Only include what appears in 3 or more conversations OR is
  explicitly high signal (named project, strong correction,
  distinctive phrase)
- Never use placeholder language — if you don't have real content,
  omit the section entirely
- Never list the AI's responses — only what the human said and did
- Write WHO and WHAT summaries as 2-3 sentence narratives, not lists
- Keep the total document under 500 words

Use this exact structure:

## Who
[2-3 sentence narrative: who this person is, what they do,
what drives them. Use their own words where possible.]

**Name:** {name}
**Role:** [most frequent or most specific role signal]
**Based in:** [location if explicitly stated, otherwise omit]
**Credentials:** [top 3-5 only — companies, patents, civic roles]

## What They Work On
[2-3 sentence narrative: their main domains and current focus]

**Active Projects:**
[List only projects appearing in 2+ conversations, with one-line
outcome. Maximum 5.]

**Open Threads:**
[List only unresolved questions or next steps. Maximum 3.]

## How To Work With Them
[2-3 sentence narrative: communication style, pace, what they value]

**Say this, not that:**
[Top 3-5 correction patterns verbatim — what they pushed back on
and what they wanted instead. Format: "not X → Y"]

**Their words:**
[Top 5 verbatim phrases that best capture their voice.
Must be the human's words, never the AI's.
Prefer distinctive framings over generic statements.]

## Don't Do This
[Bullet list of 3-5 explicit negative preferences extracted from
corrections — things they have pushed back on repeatedly.
Examples: no em dashes, no bullet points, less formal, etc.]

Extractions:
{extractions_json}"""

_VALID_BUCKETS = {
    "Work & Career", "Creative Projects", "Technical & Coding",
    "Research & Learning", "Health & Wellness", "Personal & Family",
    "Finance & Legal", "Hobbies & Interests", "Travel & Planning", "General",
}

# ── Aggregation filters ───────────────────────────────────────────────────────

_PERSONAL_SKIP_VALUES = frozenset({
    "null", "none", "n/a", "not mentioned", "not applicable",
    "unknown", "not specified", "n/a.",
})
_PERSONAL_NOISE_TERMS = frozenset({"feminine", "diaper", "pad"})

# ── FIX 2: Genealogy / third-party heritage noise filter ─────────────────────
# These signal the personal context is about someone being *researched*,
# not self-disclosed by the human.
_PERSONAL_GENEALOGY_TERMS = frozenset({
    "born in", "village", "province", "guangdong", "canton", "fujian",
    "ancestry", "heritage", "genealogy", "passed away", "who passed",
    "emigrated", "immigrated", "maiden name", "née", "baptized",
    "christened", "buried", "interred", "death certificate",
    "birth certificate", "immigration record", "ship manifest",
    "taishan", "lishan", "ancestral", "hometown",
})

_CRED_NOISE_RE = re.compile(
    r'\b\d{3,}\b'
    r'|\b(?:chrome|firefox|safari|edge|opera|brave)\b'
    r'|\b(?:windows\s?\d*|macos|mac\s?os(?:\s?x)?|ubuntu|debian|linux|android|ios)\b'
    r'|\b\d+\s*(?:gb|mb|tb|ghz|mhz)\b'
    r'|\b(?:inurl|intitle|site|filetype):'
    , re.IGNORECASE
)

_PROJECT_NOISE = frozenset({"null", "none", "none mentioned", "not mentioned", "n/a"})

_PERSONAL_SKIP_EXACT = frozenset({
    "married", "children", "neighbor", "location not specified",
    "no personal context shared", "no personal context",
    "no personal context mentioned", "location unknown",
    "silicon valley",
})

_PERSONAL_NAME_RE = re.compile(r'^[A-Z][a-z]+(?: [A-Z][a-z]+)?$')

_PERSONAL_BEHAVIORAL_TERMS = frozenset({
    "slowdown", "double-check", "double check", "assumptions", "inferences",
    "slow down",
})

_PERSONAL_HARDWARE_RE = re.compile(
    r'\b(?:dell|xps|thinkpad|latitude|inspiron|pavilion|macbook|surface|lenovo|hp\s+\w+)\b'
    r'|\b\d{4,}\b',
    re.IGNORECASE
)

# ── Email noise filter ────────────────────────────────────────────────────────
_EMAIL_RE = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}')


def _call_ollama(prompt: str, model: str = DEFAULT_MODEL, system: str = "") -> Optional[str]:
    """Make a request to the local Ollama API."""
    body: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
    if system:
        body["system"] = system
    payload = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("response", "")
    except urllib.error.URLError as e:
        print(f"Ollama connection error: {e}")
        print("Is Ollama running? Start it with: ollama serve")
        return None
    except Exception as e:
        print(f"Ollama error: {e}")
        return None


def _build_conversation_texts(convo: Dict[str, Any], max_chars: int = 6000) -> tuple:
    messages = convo.get("messages", [])
    if not messages:
        return "", ""

    full_lines = []
    human_lines = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        snippet = msg["text"][:500]
        full_lines.append(f"{role}: {snippet}")
        if msg["role"] == "user":
            human_lines.append(f"User: {snippet}")

    full_text = "\n".join(full_lines)
    human_text = "\n".join(human_lines)

    if len(full_text) > max_chars:
        half = max_chars // 2
        full_text = full_text[:half] + "\n\n[... truncated ...]\n\n" + full_text[-half:]

    human_max = max_chars // 2
    if len(human_text) > human_max:
        half = human_max // 2
        human_text = human_text[:half] + "\n\n[... truncated ...]\n\n" + human_text[-half:]

    return full_text, human_text


def _repair_json(raw: str) -> str:
    if not raw.startswith("{"):
        idx = raw.find("{")
        if idx == -1:
            return raw
        raw = raw[idx:]

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
                j = i + 1
                while j < n and raw[j] in ' \t\r\n':
                    j += 1
                next_char = raw[j] if j < n else ''
                if next_char in ',}]:"' or j >= n:
                    in_string = False
        i += 1

    if in_string:
        raw += '"'

    open_brackets = raw.count('[') - raw.count(']')
    open_braces = raw.count('{') - raw.count('}')

    raw += ']' * max(0, open_brackets)
    raw += '}' * max(0, open_braces)

    return raw


def _is_prompt_leaked(text: str) -> bool:
    return "sentence" in text or "Include:" in text


def _parse_raw_response(raw: str) -> Optional[Dict[str, Any]]:
    clean = re.sub(r'^```json\s*|\s*```$', '', raw.strip())
    clean = clean.strip()
    clean = _repair_json(clean)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json
            clean = repair_json(clean)
        except ImportError:
            pass
        return json.loads(clean)


def summarize_conversation(
    convo: Dict[str, Any],
    model: str = DEFAULT_MODEL,
    summary_style: str = "concise",
) -> Optional[Dict[str, Any]]:
    title = convo.get("title", "Untitled")
    full_text, human_text = _build_conversation_texts(convo)

    if not full_text.strip():
        return None

    full_conversation = f"Title: {title}\n\n{full_text}"
    human_messages_only = f"Title: {title}\n\n{human_text}" if human_text.strip() else "(none)"
    user_prompt = _EXTRACTION_PASS.format(
        full_conversation=full_conversation,
        human_messages_only=human_messages_only,
    )

    raw = _call_ollama(user_prompt, model=model, system=PRIME_PROMPT)

    if not raw:
        return None

    if _is_prompt_leaked(raw):
        print(f"Prompt leakage detected for '{title}', using fallback extraction.")
        result = {
            "tier": "review",
            "who": {"role": None, "expertise": [], "credentials": [], "personal": []},
            "what": {"topic": title[:40], "depth": "", "project": None, "outcome": title, "open_thread": None},
            "how": {"initial_prompt": "", "corrections": [], "your_words": []},
        }
    else:
        try:
            result = _parse_raw_response(raw)
        except json.JSONDecodeError as e:
            print(f"JSON parse failed for '{title}': {e}")
            print(f"Raw response snippet: {raw[:200]}")
            result = {
                "tier": "review",
                "who": {"role": None, "expertise": [], "credentials": [], "personal": []},
                "what": {"topic": title, "depth": "", "project": None, "outcome": "Could not parse", "open_thread": None},
                "how": {"initial_prompt": "", "corrections": [], "your_words": []},
            }

    # ── FIX 1: Normalize top-level keys to lowercase ──────────────────────────
    # Llama sometimes returns WHO/WHAT/HOW in uppercase. The aggregation loop
    # in synthesize_biography() expects lowercase keys. This normalizes all
    # top-level keys so who/what/how/tier/bucket are always lowercase.
    # Guard against model returning a list instead of an object.
    if isinstance(result, list):
        result = result[0] if result else {}
    if not isinstance(result, dict):
        result = {}
    result = {k.lower(): v for k, v in result.items()}

    # Also normalize nested keys in who/what/how blocks
    for block in ("who", "what", "how"):
        if isinstance(result.get(block), dict):
            result[block] = {k.lower(): v for k, v in result[block].items()}

    result["bucket"] = "General"

    # Rule-based tier scoring — replaces LLM tier classification
    score = 0
    who_block = result.get("who") or {}
    what_block = result.get("what") or {}
    how_block = result.get("how") or {}

    if isinstance(who_block.get("role"), str) and len(who_block["role"].strip()) > 3:
        score += 2
    if isinstance(who_block.get("expertise"), list) and len(who_block["expertise"]) > 0:
        score += 2
    if isinstance(what_block.get("project"), str) and len(what_block["project"].strip()) > 3:
        score += 2
    if isinstance(what_block.get("open_thread"), str) and len(what_block["open_thread"].strip()) > 3:
        score += 1
    your_words = (how_block.get("your_words") or "")
    if isinstance(your_words, str) and len(your_words.strip()) > 10:
        score += 1
    if isinstance(who_block.get("personal"), list) and len(who_block["personal"]) > 0:
        score += 1

    if score >= 3:
        result["tier"] = "signal"
    elif score >= 1:
        result["tier"] = "review"
    else:
        result["tier"] = "noise"

    # Legacy compatibility fields
    result["title"] = title
    result["conversation_id"] = convo.get("id", "")
    result["created"] = convo.get("created", "")
    result["updated"] = convo.get("updated", "")
    result["summary"] = (result.get("what") or {}).get("outcome") or ""
    result["topics"] = [(result.get("what") or {}).get("topic")] if (result.get("what") or {}).get("topic") else []
    open_thread = (result.get("what") or {}).get("open_thread")
    result["open_threads"] = [open_thread] if open_thread else []
    result["key_decisions"] = []
    result["artifacts"] = []
    result["preferences"] = (result.get("how") or {}).get("corrections", [])

    return result


def synthesize_biography(
    extractions: list,
    output_dir,
    model: str = DEFAULT_MODEL,
    user_name: str = None,
) -> Optional[Dict[str, Any]]:
    from collections import Counter

    print("[bio] synthesize_biography() called")
    if not extractions:
        return None

    expertise_counter: Counter = Counter()
    credentials_counter: Counter = Counter()
    your_words_list: list = []
    corrections_list: list = []
    personal_list: list = []
    projects_map: dict = {}
    open_threads_list: list = []
    roles: list = []
    tier_counts = {"signal": 0, "review": 0, "noise": 0}

    _FIRST_PERSON_MARKERS = ("my name is", "i am", "i'm", "i've been", "my first name")
    _CRED_ENTITY_BLOCKLIST = [
        "axios", "fetch.ai", "columbia university",
        "mayo clinic", "reuters", "techcrunch", "wired", "forbes",
        "harvard", "mit", "stanford", "openai", "anthropic",
        "google", "microsoft", "apple", "amazon", "meta",
    ]
    _YOUR_WORDS_SKIP_TERMS = ["diaper", "pad", "feminine", "menstrual", "period"]
    _YOUR_WORDS_SKIP_PREFIXES = ("how many", "what is a")

    if extractions and isinstance(extractions[0], dict):
        print(f"[bio] first summary keys: {list(extractions[0].keys())}")
    all_tier_values = set()
    for ex in extractions:
        if isinstance(ex, dict):
            all_tier_values.add(ex.get("tier"))
    print(f"[bio] unique tier values found: {all_tier_values}")
    print(f"[bio] first 5 tier values: {[ex.get('tier') for ex in extractions[:5] if isinstance(ex, dict)]}")

    for ex in extractions:
        if not isinstance(ex, dict):
            continue

        tier_raw = ex.get("tier") or "review"
        tier = tier_raw.strip().lower() if isinstance(tier_raw, str) else "review"
        if tier in tier_counts:
            tier_counts[tier] += 1

        who = ex.get("who") or {}
        if not isinstance(who, dict):
            who = {}

        for e in (who.get("expertise") if isinstance(who.get("expertise"), list) else []):
            if isinstance(e, str) and e.strip():
                expertise_counter[e.strip()] += 1

        if tier == "signal":
            for c in (who.get("credentials") if isinstance(who.get("credentials"), list) else []):
                if not isinstance(c, str):
                    continue
                c = c.strip()
                if not c or _CRED_NOISE_RE.search(c):
                    continue
                if any(bl in c.lower() for bl in _CRED_ENTITY_BLOCKLIST):
                    continue
                credentials_counter[c] += 1

        # ── FIX 2: Personal context — filter genealogy / third-party heritage ─
        for p in (who.get("personal") if isinstance(who.get("personal"), list) else []):
            if not isinstance(p, str):
                continue
            p = p.strip()
            if not p or p.lower() in _PERSONAL_SKIP_VALUES:
                continue
            if p.lower() in _PERSONAL_SKIP_EXACT:
                continue
            if len(p) < 5 or len(p) > 100:
                continue
            if any(noise in p.lower() for noise in _PERSONAL_NOISE_TERMS):
                continue
            # Filter genealogy / heritage research misattributions
            if any(term in p.lower() for term in _PERSONAL_GENEALOGY_TERMS):
                continue
            # Filter email addresses
            if _EMAIL_RE.search(p):
                continue
            if re.search(r'\b(?:husband|wife|spouse|partner)\s+of\b', p, re.IGNORECASE):
                continue
            if any(term in p.lower() for term in _PERSONAL_BEHAVIORAL_TERMS):
                continue
            if _PERSONAL_NAME_RE.match(p):
                continue
            if _PERSONAL_HARDWARE_RE.search(p):
                continue
            personal_list.append(p)

        role = who.get("role")
        if isinstance(role, str) and role.strip():
            roles.append(role.strip())

        how = ex.get("how") or {}
        if not isinstance(how, dict):
            how = {}

        if tier == "signal":
            for w in (how.get("your_words") if isinstance(how.get("your_words"), list) else []):
                if not isinstance(w, str):
                    continue
                w = w.strip()
                if not w:
                    continue
                if w.endswith("?"):
                    continue
                if len(w.split()) < 4:
                    continue
                if any(t in w.lower() for t in _YOUR_WORDS_SKIP_TERMS):
                    continue
                if w.lower().startswith(_YOUR_WORDS_SKIP_PREFIXES):
                    continue
                your_words_list.append(w)

        for c in (how.get("corrections") if isinstance(how.get("corrections"), list) else []):
            if isinstance(c, str) and c.strip():
                corrections_list.append(c.strip())

        what = ex.get("what") or {}
        if not isinstance(what, dict):
            what = {}

        open_thread = what.get("open_thread")
        if isinstance(open_thread, str) and open_thread.strip():
            open_threads_list.append(open_thread.strip())

        proj_name = what.get("project")
        if isinstance(proj_name, str) and proj_name.strip():
            pname = proj_name.strip()
            if pname.lower() not in _PROJECT_NOISE:
                if pname not in projects_map:
                    projects_map[pname] = {
                        "name": pname,
                        "description": what.get("outcome") or "",
                        "status": what.get("outcome") or "open",
                        "conversation_count": 0,
                    }
                projects_map[pname]["conversation_count"] += 1

    # Personal context dedup + cap
    _seen_lower: dict = {}
    for _p in personal_list:
        _key = _p.lower()
        if _key not in _seen_lower:
            _seen_lower[_key] = _p
    _unique_personal = list(_seen_lower.values())

    _deduped_personal: list = []
    for _p in _unique_personal:
        _dominated = any(
            _p != _other and _p.lower() in _other.lower()
            for _other in _unique_personal
        )
        if not _dominated:
            _deduped_personal.append(_p)

    _PRIORITY_TERMS = frozenset({
        "philadelphia", "metro", "area", "years old", "age",
        "health", "medical", "condition", "email", "@",
        "stage", "life", "city", "county", "state",
    })

    def _personal_priority(item: str) -> int:
        low = item.lower()
        return sum(1 for t in _PRIORITY_TERMS if t in low)

    _deduped_personal.sort(key=_personal_priority, reverse=True)
    personal_list = _deduped_personal[:10]

    _LEAKAGE_PHRASES = [
        "domains the human", "they taught or led", "not asked",
        "topics the human", "follow-up questions", "pattern recognition",
        "topics where the human", "came for assistance", "not expertise",
        "not necessarily interest", "just needed help", "none mentioned",
        "none specified", "null", "none", "n/a"
    ]

    expertise_counter = Counter({
        k: v for k, v in expertise_counter.items()
        if not any(phrase in k.lower() for phrase in _LEAKAGE_PHRASES)
        and len(k) < 60
    })

    top_expertise = [e for e, _ in expertise_counter.most_common(10)]
    most_common_role = Counter(roles).most_common(1)[0][0] if roles else None
    projects_list = sorted(projects_map.values(), key=lambda p: -p["conversation_count"])[:10]

    detected_name = user_name.strip() if user_name and user_name.strip() else "User"
    print(f"[bio] detected name: {detected_name}")

    raw_creds_ordered = [c for c, _ in credentials_counter.most_common()]
    deduped_creds: list = []
    for cred in raw_creds_ordered:
        dominated = any(
            cred != other and cred.lower() in other.lower()
            for other in raw_creds_ordered
        )
        if not dominated:
            deduped_creds.append(cred)
    credentials_list = deduped_creds[:15]

    aggregated_profile = {
        "conversation_count": len(extractions),
        "tier_counts": tier_counts,
        "name": detected_name,
        "most_common_role": most_common_role,
        "detected_name": detected_name,
        "top_expertise": top_expertise,
        "credentials": credentials_list,
        "top_projects": [p["name"] for p in projects_list[:5]],
        "sample_your_words": your_words_list[:10],
        "sample_corrections": corrections_list[:10],
    }

    agg_json = json.dumps(aggregated_profile, indent=2)
    user_prompt = _SYNTHESIS_PASS_V2.format(
        n=len(extractions),
        name=detected_name,
        extractions_json=agg_json,
    )

    synthesis_system = f"{PRIME_PROMPT}\n\nYou are a helpful assistant. Write clean markdown."
    raw = _call_ollama(user_prompt, model=model, system=synthesis_system)
    print(f"[bio] raw response length: {len(raw) if raw else 0}")
    if not raw:
        print("Synthesis pass returned no response.")
        return None

    # New synthesis path — prompt returns markdown directly
    raw_markdown = raw.strip()

    # Write markdown directly to get_to_know_me.md
    bio_path = os.path.join(output_dir, "get_to_know_me.md")
    with open(bio_path, "w", encoding="utf-8") as f:
        f.write(raw_markdown)
    print(f"Wrote biography to {bio_path}")

    return {"bio_path": bio_path, "markdown": raw_markdown}


def summarize_all(
    conversations: list,
    model: str = DEFAULT_MODEL,
    progress_callback=None,
    summary_style: str = "concise",
) -> list:
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
    total = len(conversations)
    for i, convo in enumerate(conversations):
        title = convo.get("title", "Untitled")
        summary = summarize_conversation(convo, model=model, summary_style=summary_style)
        yield (i + 1, total, title, summary)


def check_ollama_running() -> bool:
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/tags",
            method="GET"
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False