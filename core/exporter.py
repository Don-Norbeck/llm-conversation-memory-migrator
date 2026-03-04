"""
core/exporter.py
Generates clean markdown context files from classified summaries.
Produces per-bucket files and a master context document.
"""

import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional


def sanitize_filename(name: str) -> str:
    """Convert a bucket name to a safe filename."""
    clean = re.sub(r'[\\/*?:"<>|&]', "_", name)
    clean = clean.replace(" ", "_")
    return clean.strip("_")


def format_summary_as_markdown(summary: Dict[str, Any]) -> str:
    """Format a single conversation summary as a markdown section."""
    lines = []

    title = summary.get("title", "Untitled")
    created = summary.get("created", "unknown")
    updated = summary.get("updated", "unknown")

    lines.append(f"### {title}")
    lines.append(f"*Created: {created} | Updated: {updated}*")
    lines.append("")

    if summary.get("summary"):
        lines.append(summary["summary"])
        lines.append("")

    decisions = summary.get("key_decisions", [])
    if decisions and isinstance(decisions, list):
        lines.append("**Key Decisions:**")
        for d in decisions:
            lines.append(f"- {d}")
        lines.append("")

    artifacts = summary.get("artifacts", [])
    if artifacts and isinstance(artifacts, list):
        lines.append("**Artifacts Created:**")
        for a in artifacts:
            lines.append(f"- {a}")
        lines.append("")

    threads = summary.get("open_threads", [])
    if threads and isinstance(threads, list):
        lines.append("**Open Threads:**")
        for t in threads:
            lines.append(f"- {t}")
        lines.append("")

    prefs = summary.get("preferences", [])
    if prefs and isinstance(prefs, list):
        lines.append("**Preferences & Style Notes:**")
        for p in prefs:
            lines.append(f"- {p}")
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

    lines.append(f"# {bucket_name} — Context Document")
    lines.append(f"*Generated: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*")
    lines.append(f"*Conversations: {len(summaries)}*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Overview — collect all topics
    all_topics = []
    for s in summaries:
        topics = s.get("topics", [])
        if isinstance(topics, list):
            all_topics.extend(topics)
    unique_topics = sorted(set(all_topics))
    if unique_topics:
        lines.append("## Topic Overview")
        lines.append(", ".join(unique_topics))
        lines.append("")

    # All preferences across bucket
    all_prefs = []
    for s in summaries:
        prefs = s.get("preferences", [])
        if isinstance(prefs, list):
            all_prefs.extend(prefs)
    if all_prefs:
        lines.append("## Accumulated Preferences & Style Notes")
        for p in set(all_prefs):
            lines.append(f"- {p}")
        lines.append("")

    # All open threads
    all_threads = []
    for s in summaries:
        threads = s.get("open_threads", [])
        if isinstance(threads, list):
            all_threads.extend(threads)
    if all_threads:
        lines.append("## Open Threads & Next Steps")
        for t in all_threads:
            lines.append(f"- {t}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Conversation Summaries")
    lines.append("")

    sorted_summaries = sorted(summaries, key=lambda x: x.get("created", ""))
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
    output_path = output_dir / "master_context.md"
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

        all_decisions = []
        for s in summaries:
            decisions = s.get("key_decisions", [])
            if isinstance(decisions, list):
                all_decisions.extend(decisions)
        if all_decisions:
            lines.append("**Key Decisions & Conclusions:**")
            for d in all_decisions[:10]:
                lines.append(f"- {d}")
            lines.append("")

        all_prefs = []
        for s in summaries:
            prefs = s.get("preferences", [])
            if isinstance(prefs, list):
                all_prefs.extend(prefs)
        if all_prefs:
            lines.append("**Preferences & Style:**")
            for p in list(set(all_prefs))[:5]:
                lines.append(f"- {p}")
            lines.append("")

        all_threads = []
        for s in summaries:
            threads = s.get("open_threads", [])
            if isinstance(threads, list):
                all_threads.extend(threads)
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

    for bucket, summaries in grouped.items():
        if not summaries:
            continue
        if not isinstance(summaries, list):
            continue
        path = export_bucket(bucket, summaries, output_dir)
        exported[bucket] = path
        print(f"  Exported: {path.name} ({len(summaries)} conversations)")

    master_path = export_master_context(grouped, output_dir, user_name)
    exported["__master__"] = master_path
    print(f"  Exported: {master_path.name}")

    print(f"\nTotal files written: {len(exported)}")
    return exported