"""
adapters/chatgpt.py
Parses ChatGPT export archives into a normalized conversation format.
Cleans up temporary files after parsing to protect user privacy.
"""

import json
import zipfile
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional


def ts_to_str(ts) -> str:
    """Convert a Unix timestamp to a readable UTC string."""
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return "unknown date"


def get_message_text(message: Dict) -> str:
    """Extract plain text from a ChatGPT message object."""
    try:
        parts = message["content"]["parts"]
        return " ".join(str(p) for p in parts if isinstance(p, str))
    except Exception:
        return ""


def parse_conversation(convo: Dict) -> Dict[str, Any]:
    """
    Normalize a single ChatGPT conversation into a standard format.

    Returns:
        {
            "id": str,
            "title": str,
            "created": str,
            "updated": str,
            "messages": [{"role": str, "text": str, "timestamp": str}]
        }
    """
    mapping = convo.get("mapping", {})
    messages = []

    for node in mapping.values():
        msg = node.get("message")
        if not msg:
            continue
        role = msg.get("author", {}).get("role", "unknown")
        if role not in ("user", "assistant"):
            continue
        ts = msg.get("create_time") or 0
        text = get_message_text(msg)
        if text.strip():
            messages.append({
                "role": role,
                "text": text.strip(),
                "timestamp": ts_to_str(ts),
                "ts_raw": float(ts) if ts else 0
            })

    messages.sort(key=lambda x: x["ts_raw"])

    return {
        "id": convo.get("id", ""),
        "title": convo.get("title", "Untitled"),
        "created": ts_to_str(convo.get("create_time")),
        "updated": ts_to_str(convo.get("update_time")),
        "messages": messages
    }


def extract_zip(zip_path: Path) -> Path:
    """
    Extract only JSON files from a ChatGPT export zip.
    Returns the path to the extraction directory.
    """
    extract_dir = zip_path.parent / "chatgpt_extracted"
    existing = list(extract_dir.glob("conversations*.json"))
    if not existing:
        print(f"Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                try:
                    if member.filename.endswith(".json"):
                        zf.extract(member, extract_dir)
                except Exception as e:
                    print(f"Skipping {member.filename}: {e}")
        print("Extraction complete.")
    else:
        print("Using existing extracted files.")
    return extract_dir


def load_conversations(extract_dir: Path) -> List[Dict]:
    """
    Load and merge all conversations-*.json files from an extracted export.
    Returns a flat list of raw conversation dicts.
    """
    json_files = sorted(extract_dir.glob("conversations*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No conversations*.json found in {extract_dir}\n"
            f"Make sure you uploaded a valid ChatGPT export zip."
        )

    all_convos = []
    for jf in json_files:
        print(f"  Loading {jf.name} ({jf.stat().st_size / 1e6:.1f} MB) ...")
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            all_convos.extend(data)
        else:
            all_convos.append(data)

    print(f"Total conversations loaded: {len(all_convos)}")
    return all_convos


def cleanup_extracted(extract_dir: Path) -> None:
    """
    Delete extracted files from temp directory after parsing.
    Protects user privacy by not leaving conversation data on disk.
    """
    try:
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
            print(f"🧹 Cleaned up temporary files: {extract_dir}")
    except Exception as e:
        print(f"Warning: Could not clean up temp files at {extract_dir}: {e}")
        print("You may want to manually delete this folder to protect your privacy.")


def load_from_zip(zip_path: Path) -> List[Dict[str, Any]]:
    """
    Main entry point. Given a path to a ChatGPT export zip,
    returns a list of normalized conversation dicts.
    Cleans up extracted files after parsing.
    """
    extract_dir = None
    try:
        extract_dir = extract_zip(zip_path)
        raw_convos = load_conversations(extract_dir)
        print("Parsing conversations ...")
        parsed = [parse_conversation(c) for c in raw_convos]
        print(f"Parsed {len(parsed)} conversations.")
        return parsed
    finally:
        # Always clean up — even if parsing fails
        if extract_dir:
            cleanup_extracted(extract_dir)


def conversation_to_markdown(convo: Dict[str, Any]) -> str:
    """Convert a normalized conversation to a Markdown string."""
    lines = [
        f"# {convo['title']}",
        f"**Created:** {convo['created']}  |  **Updated:** {convo['updated']}",
        "---",
        ""
    ]
    for msg in convo["messages"]:
        label = "**You:**" if msg["role"] == "user" else "**Assistant:**"
        lines.append(f"{label} *({msg['timestamp']})*")
        lines.append(msg["text"])
        lines.append("")
    return "\n".join(lines)