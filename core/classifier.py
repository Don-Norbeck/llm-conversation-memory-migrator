"""
core/classifier.py
Groups summarized conversations into topic buckets.
Supports both AI-assigned buckets and manual override.
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict


# ── Default bucket definitions ───────────────────────────────────────────────

DEFAULT_BUCKETS = [
    "Work & Career",
    "Creative Projects",
    "Technical & Coding",
    "Research & Learning",
    "Health & Wellness",
    "Personal & Family",
    "Finance & Legal",
    "Hobbies & Interests",
    "Travel & Planning",
    "General",
]


def classify_summaries(
    summaries: List[Dict[str, Any]],
    custom_buckets: Optional[List[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group summaries by their AI-assigned bucket.

    Args:
        summaries: List of summary dicts from summarizer.py
        custom_buckets: Optional list of bucket names to use
                        (summaries with unknown buckets go to General)

    Returns:
        Dict mapping bucket name -> list of summaries
    """
    buckets = custom_buckets or DEFAULT_BUCKETS
    bucket_set = set(buckets)
    grouped = defaultdict(list)

    for summary in summaries:
        bucket = summary.get("bucket", "General")
        # Normalize — if model returned unknown bucket, send to General
        if bucket not in bucket_set:
            bucket = "General"
        grouped[bucket].append(summary)

    # Ensure all buckets exist even if empty
    for bucket in buckets:
        if bucket not in grouped:
            grouped[bucket] = []

    return dict(grouped)


def get_bucket_stats(grouped: Dict[str, List]) -> List[Dict]:
    """
    Return a sorted list of bucket statistics for display in UI.

    Returns:
        List of {"bucket": str, "count": int, "conversations": [...]}
    """
    stats = []
    for bucket, summaries in grouped.items():
        if summaries:  # Only include non-empty buckets
            stats.append({
                "bucket": bucket,
                "count": len(summaries),
                "conversations": summaries
            })
    return sorted(stats, key=lambda x: x["count"], reverse=True)


def reassign_conversation(
    grouped: Dict[str, List],
    conversation_id: str,
    new_bucket: str
) -> Dict[str, List]:
    """
    Move a conversation from its current bucket to a new one.
    Used for manual override in the review UI.

    Args:
        grouped: Current bucket groupings
        conversation_id: ID of conversation to move
        new_bucket: Target bucket name

    Returns:
        Updated grouped dict
    """
    target_summary = None

    # Find and remove from current bucket
    for bucket, summaries in grouped.items():
        for summary in summaries:
            if summary.get("conversation_id") == conversation_id:
                target_summary = summary
                grouped[bucket] = [
                    s for s in summaries
                    if s.get("conversation_id") != conversation_id
                ]
                break
        if target_summary:
            break

    if target_summary:
        if new_bucket not in grouped:
            grouped[new_bucket] = []
        target_summary["bucket"] = new_bucket
        grouped[new_bucket].append(target_summary)

    return grouped


def merge_buckets(
    grouped: Dict[str, List],
    source_bucket: str,
    target_bucket: str
) -> Dict[str, List]:
    """
    Merge all conversations from source_bucket into target_bucket.
    Used when user wants to combine two topics.

    Args:
        grouped: Current bucket groupings
        source_bucket: Bucket to merge from (will be emptied)
        target_bucket: Bucket to merge into

    Returns:
        Updated grouped dict
    """
    if source_bucket not in grouped:
        return grouped

    if target_bucket not in grouped:
        grouped[target_bucket] = []

    for summary in grouped[source_bucket]:
        summary["bucket"] = target_bucket

    grouped[target_bucket].extend(grouped[source_bucket])
    grouped[source_bucket] = []

    return grouped


def rename_bucket(
    grouped: Dict[str, List],
    old_name: str,
    new_name: str
) -> Dict[str, List]:
    """
    Rename a bucket.

    Args:
        grouped: Current bucket groupings
        old_name: Current bucket name
        new_name: New bucket name

    Returns:
        Updated grouped dict
    """
    if old_name not in grouped:
        return grouped

    grouped[new_name] = grouped.pop(old_name)
    for summary in grouped[new_name]:
        summary["bucket"] = new_name

    return grouped


def add_bucket(
    grouped: Dict[str, List],
    bucket_name: str
) -> Dict[str, List]:
    """Add a new empty bucket."""
    if bucket_name not in grouped:
        grouped[bucket_name] = []
    return grouped


def get_all_bucket_names(grouped: Dict[str, List]) -> List[str]:
    """Return sorted list of all non-empty bucket names."""
    return sorted([b for b, s in grouped.items() if s])