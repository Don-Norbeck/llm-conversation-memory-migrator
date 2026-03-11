"""
core/exporter.py
Generates clean markdown context files from classified summaries.
Produces per-bucket files and a master context document.
"""

import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional


def _count_items(items: list) -> Dict[str, int]:
    """Return a dict mapping each item to its occurrence count."""
    counts: Dict[str, int] = {}
    for item in items:
        if item:
            counts[item] = counts.get(item, 0) + 1
    return counts


def _safe_join(value, sep: str = ", ") -> str:
    """Join a value as strings. Coerces non-list to empty; coerces items to str."""
    if not isinstance(value, list):
        return ""
    return sep.join(str(x) for x in value if x is not None)


def sanitize_filename(name: str) -> str:
    """Convert a bucket name to a safe filename."""
    clean = name.replace("&", "and")
    clean = re.sub(r'[^a-zA-Z0-9]+', "_", clean)
    return clean.strip("_")


def format_summary_as_markdown(summary: Dict[str, Any]) -> str:
    """Format a single conversation summary as a markdown section."""
    lines = []

    title = summary.get("title", "Untitled")
    created = summary.get("created", "unknown")
    updated = summary.get("updated", "unknown")
    tier = summary.get("tier", "")

    tier_str = f" | tier: {tier}" if tier else ""
    lines.append(f"### {title}")
    lines.append(f"*Created: {created} | Updated: {updated}{tier_str}*")
    lines.append("")

    # WHAT
    what = summary.get("what") or {}
    outcome = what.get("outcome")
    if outcome:
        lines.append(outcome)
        lines.append("")

    if what.get("depth"):
        lines.append(f"**Depth:** {what['depth']}")
        lines.append("")

    open_thread = what.get("open_thread")
    if open_thread:
        lines.append(f"**Open thread:** {open_thread}")
        lines.append("")

    # WHO
    who = summary.get("who") or {}
    who_lines = []
    if who.get("role"):
        who_lines.append(f"- **Role:** {who['role']}")
    if who.get("expertise"):
        who_lines.append(f"- **Expertise:** {_safe_join(who['expertise'])}")
    if who.get("credentials"):
        who_lines.append(f"- **Credentials:** {_safe_join(who['credentials'])}")
    if who.get("personal"):
        who_lines.append(f"- **Personal:** {_safe_join(who['personal'])}")
    if who_lines:
        lines.append("**WHO:**")
        lines.extend(who_lines)
        lines.append("")

    # HOW
    how = summary.get("how") or {}
    how_lines = []
    if how.get("initial_prompt"):
        how_lines.append(f"- **Opening ask:** {how['initial_prompt']}")
    if how.get("corrections"):
        how_lines.append(f"- **Corrections:** {_safe_join(how['corrections'])}")
    if how.get("your_words"):
        words = how["your_words"] if isinstance(how["your_words"], list) else []
        quoted = " ".join(f'"{str(w)}"' for w in words if w is not None)
        how_lines.append(f"- **Your words:** {quoted}")
    if how_lines:
        lines.append("**HOW:**")
        lines.extend(how_lines)
        lines.append("")

    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def export_bucket(
    bucket_name: str,
    summaries: List[Dict[str, Any]],
    output_dir: Path
) -> Path:
    """
    Export a single bucket as a markdown context file.
    Returns the path to the written file.
    """
    filename = sanitize_filename(bucket_name) + "_context.md"
    output_path = output_dir / filename

    lines = []

    word_count = sum(
        len(format_summary_as_markdown(s).split()) for s in summaries
    )
    lines.append(f"# {bucket_name} — Context Document")
    lines.append(f"*Generated: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*")
    convo_word = "conversation" if len(summaries) == 1 else "conversations"
    lines.append(f"*{len(summaries)} {convo_word}*")
    if word_count > 5000:
        lines.append("")
        lines.append(f"> **Note:** This bucket contains {len(summaries)} conversations. Consider splitting into sub-topics.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Overview — collect all topics
    all_topics = []
    for s in summaries:
        topic = (s.get("what") or {}).get("topic")
        if topic:
            all_topics.append(topic)
    unique_topics = sorted(set(all_topics))
    if unique_topics:
        lines.append("## Topic Overview")
        lines.append(", ".join(unique_topics))
        lines.append("")

    # Expertise signals
    all_expertise = []
    for s in summaries:
        all_expertise.extend((s.get("who") or {}).get("expertise") or [])
    if all_expertise:
        lines.append("## Expertise Signals")
        for e in sorted(set(all_expertise)):
            lines.append(f"- {e}")
        lines.append("")

    # Correction patterns
    all_corrections = []
    for s in summaries:
        all_corrections.extend((s.get("how") or {}).get("corrections") or [])
    if all_corrections:
        lines.append("## Correction Patterns")
        for c in sorted(set(all_corrections)):
            lines.append(f"- {c}")
        lines.append("")

    # Open threads
    all_threads = []
    for s in summaries:
        thread = (s.get("what") or {}).get("open_thread")
        if thread:
            all_threads.append(thread)
    if all_threads:
        lines.append("## Open Threads & Next Steps")
        for t in all_threads:
            lines.append(f"- {t}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Conversation Summaries")
    lines.append("")

    sorted_summaries = sorted(summaries, key=lambda x: str(x.get("created") or ""))
    for summary in sorted_summaries:
        lines.append(format_summary_as_markdown(summary))

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def export_master_context(
    grouped: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
    user_name: Optional[str] = None
) -> Path:
    """
    Generate a master context document summarizing all buckets.
    Returns the path to the written file.
    """
    output_path = output_dir / "get_to_know_me_full.md"
    lines = []

    name_str = f" — {user_name}" if user_name else ""
    lines.append(f"# Master Context Document{name_str}")
    lines.append(f"*Generated: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*")
    lines.append("")
    lines.append("This document summarizes accumulated context, preferences, and")
    lines.append("active projects from previous AI conversations.")
    lines.append("Upload this to your new LLM service to restore context.")
    lines.append("")
    lines.append("---")
    lines.append("")

    total = sum(len(s) for s in grouped.values() if isinstance(s, list))
    active_buckets = [b for b, s in grouped.items() if s and isinstance(s, list)]
    lines.append("## Overview")
    lines.append(f"- **Total conversations analyzed:** {total}")
    lines.append(f"- **Topic areas:** {len(active_buckets)}")
    lines.append(f"- **Active areas:** {', '.join(active_buckets)}")
    lines.append("")

    for bucket, summaries in grouped.items():
        if not summaries or not isinstance(summaries, list):
            continue

        lines.append(f"## {bucket}")
        lines.append(f"*{len(summaries)} conversations*")
        lines.append("")

        # Outcomes — up to 10
        all_outcomes = []
        for s in summaries:
            outcome = (s.get("what") or {}).get("outcome")
            if outcome:
                all_outcomes.append(outcome)
        if all_outcomes:
            lines.append("**Outcomes & Intent:**")
            for o in all_outcomes[:10]:
                lines.append(f"- {o}")
            lines.append("")

        # Expertise signals
        all_expertise = []
        for s in summaries:
            all_expertise.extend((s.get("who") or {}).get("expertise") or [])
        if all_expertise:
            lines.append("**Expertise:**")
            for e in sorted(set(all_expertise)):
                lines.append(f"- {e}")
            lines.append("")

        # Correction patterns — up to 5 unique
        all_corrections = []
        for s in summaries:
            all_corrections.extend((s.get("how") or {}).get("corrections") or [])
        if all_corrections:
            lines.append("**Correction Patterns:**")
            for c in list(set(all_corrections))[:5]:
                lines.append(f"- {c}")
            lines.append("")

        # Open threads — up to 5
        all_threads = []
        for s in summaries:
            thread = (s.get("what") or {}).get("open_thread")
            if thread:
                all_threads.append(thread)
        if all_threads:
            lines.append("**Open Threads:**")
            for t in all_threads[:5]:
                lines.append(f"- {t}")
            lines.append("")

        lines.append("---")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def export_all(
    grouped: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
    user_name: Optional[str] = None
) -> Dict[str, Path]:
    """
    Export all buckets plus master context document.
    Returns dict mapping bucket name -> output file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = {}
    # Bucket *_context.md files are disabled; bucket field is retained on extractions

    master_path = export_master_context(grouped, output_dir, user_name)
    exported["__master__"] = master_path
    print(f"  Exported: {master_path.name}")

    print(f"\nTotal files written: {len(exported)}")
    return exported