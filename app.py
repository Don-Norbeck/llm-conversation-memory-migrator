"""
app.py
Main entry point for llm-conversation-memory-migrator.
Launches a local Gradio UI — no internet connection required.
"""

import json
import os
import tempfile
import time

# ── Privacy: disable all huggingface and gradio telemetry ────────────────────
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

import gradio as gr
from pathlib import Path
from datetime import datetime

from core.pipeline import (
    stage1_ingest, stage2_badge, get_selected_conversations, apply_test_mode,
    stage4_extract_gen, stage5_classify, stage6_synthesize, stage6_export,
    check_ollama_running,
)
from core.badges import is_noise, badge_summary
from core import config

# ── State ─────────────────────────────────────────────────────────────────────

state = {
    "conversations": [],
    "summaries": [],
    "grouped": {},
    "export_dir": None,
    "analysis_complete": False,
    "ollama_model": None,
    "psychographic": {},
    "badge_summary": {},
    # Conversation browser state
    "conv_table": {},    # id -> {id, title, date_str, date_dt, word_count, selected, badges, noise}
    "visible_ids": [],   # ordered list of IDs currently shown (after sort/filter)
    "sort_by": "date",
    "from_date": "",
    "to_date": "",
}


# ── Conversation browser helpers ───────────────────────────────────────────────

def _word_count(conv) -> int:
    return sum(len(m["text"].split()) for m in conv.get("messages", []))


def _parse_date_str(date_str: str) -> datetime:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M UTC")
    except Exception:
        return datetime.min


def _build_conv_table(conversations: list) -> dict:
    table = {}
    for conv in conversations:
        badges = conv.get("badges", [])
        table[conv["id"]] = {
            "id": conv["id"],
            "title": conv["title"],
            "date_str": conv["created"],
            "date_dt": _parse_date_str(conv["created"]),
            "word_count": _word_count(conv),
            "selected": True,
            "badges": badges,
            "noise": is_noise(conv),
        }
    return table


def _apply_sort_filter() -> list:
    """Return ordered list of visible IDs based on current sort/filter state."""
    rows = list(state["conv_table"].values())

    # Date range filter
    from_date = state["from_date"].strip()
    to_date = state["to_date"].strip()
    if from_date:
        try:
            from_dt = datetime.strptime(from_date, "%Y-%m-%d")
            rows = [r for r in rows if r["date_dt"] >= from_dt]
        except ValueError:
            pass
    if to_date:
        try:
            to_dt = datetime.strptime(to_date, "%Y-%m-%d").replace(hour=23, minute=59)
            rows = [r for r in rows if r["date_dt"] <= to_dt]
        except ValueError:
            pass

    # Sort
    sort_by = state["sort_by"]
    if sort_by == "date":
        rows.sort(key=lambda r: r["date_dt"], reverse=True)
    elif sort_by == "size":
        rows.sort(key=lambda r: r["word_count"], reverse=True)
    elif sort_by == "title":
        rows.sort(key=lambda r: r["title"].lower())

    return [r["id"] for r in rows]


def _to_df_rows() -> list:
    """Convert visible_ids to list-of-lists for Gradio DataFrame."""
    rows = []
    for id_ in state["visible_ids"]:
        r = state["conv_table"][id_]
        badge_str = ", ".join(r.get("badges", [])[:3])
        rows.append([r["selected"], r["title"], r["date_str"], r["word_count"], badge_str, r.get("noise", False)])
    return rows


def _selection_count_text() -> str:
    total = len(state["conv_table"])
    if total == 0:
        return ""
    selected_total = sum(1 for r in state["conv_table"].values() if r["selected"])
    visible = len(state["visible_ids"])
    if visible == total:
        return f"**{selected_total} of {total} selected**"
    visible_selected = sum(
        1 for id_ in state["visible_ids"] if state["conv_table"][id_]["selected"]
    )
    return (
        f"**{visible_selected} of {visible} shown selected** "
        f"({selected_total} of {total} total selected)"
    )


# ── Step 1: Upload & Parse ────────────────────────────────────────────────────

def handle_upload(zip_file):
    """Returns (upload_msg, inspect_btn_update)."""
    _inspect_hidden = gr.update(visible=False)

    if zip_file is None:
        return (
            "❌ No file uploaded. Please upload your ChatGPT export zip.",
            _inspect_hidden,
        )
    try:
        tmp_path = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".zip")
            os.close(tmp_fd)
            with open(tmp_path, 'wb') as f:
                f.write(zip_file)
            ingest = stage1_ingest(tmp_path)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        if "error" in ingest:
            return (f"❌ Error loading file: {ingest['error']}", _inspect_hidden)

        badged = stage2_badge(ingest["conversations"])
        conversations = badged["conversations"]

        state["conversations"] = conversations
        state["summaries"] = []
        state["grouped"] = {}
        state["analysis_complete"] = False
        state["badge_summary"] = badged["badge_summary"]

        # Build browser state
        state["conv_table"] = _build_conv_table(conversations)
        state["sort_by"] = "date"
        state["from_date"] = ""
        state["to_date"] = ""
        state["visible_ids"] = _apply_sort_filter()

        msg = f"✅ Loaded **{len(conversations)} conversations**. Click **Inspect →** to continue."
        return (msg, gr.update(visible=True))

    except FileNotFoundError as e:
        return (f"❌ Invalid export file: {e}", _inspect_hidden)
    except Exception as e:
        return (f"❌ Error loading file: {e}", _inspect_hidden)


def _badge_dashboard_html() -> str:
    """Render the badge inspection dashboard HTML."""
    conversations = state["conversations"]
    summary = state["badge_summary"]
    total = len(conversations)
    if not total:
        return ""

    noise_count = sum(1 for c in conversations if is_noise(c))
    signal_count = total - noise_count
    noise_pct = round(noise_count / total * 100)
    signal_pct = round(signal_count / total * 100)

    cards_html = "".join(
        f'<div style="background:#161b22;border:1px solid rgba(255,107,26,0.12);'
        f'padding:9px 14px;display:flex;justify-content:space-between;align-items:center;gap:8px">'
        f'<span style="font-family:\'Space Mono\',monospace;font-size:10px;color:#e6edf3;'
        f'letter-spacing:0.04em">{name}</span>'
        f'<span style="font-family:\'Space Mono\',monospace;font-size:13px;font-weight:700;'
        f'color:#ff6b1a;flex-shrink:0">{count}</span>'
        f'</div>'
        for name, count in summary.items()
    )

    return (
        f'<div style="background:#0d1117;border:1px solid rgba(255,107,26,0.2);'
        f'padding:20px 24px;font-family:\'Space Mono\',monospace">'
        f'<div style="font-size:10px;color:#ff6b1a;letter-spacing:0.15em;'
        f'text-transform:uppercase;margin-bottom:18px">// BADGE INSPECTION REPORT</div>'
        f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:22px">'
        f'<div style="background:#161b22;border:1px solid rgba(255,107,26,0.2);padding:14px 16px">'
        f'<div style="font-size:9px;color:#8b949e;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:6px">Total</div>'
        f'<div style="font-size:30px;font-weight:700;color:#e6edf3;line-height:1">{total}</div>'
        f'<div style="font-size:9px;color:#8b949e;margin-top:3px">conversations</div>'
        f'</div>'
        f'<div style="background:#161b22;border:1px solid rgba(255,107,26,0.2);padding:14px 16px">'
        f'<div style="font-size:9px;color:#8b949e;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:6px">Noise</div>'
        f'<div style="font-size:30px;font-weight:700;color:#ff6b1a;line-height:1">{noise_count}</div>'
        f'<div style="font-size:9px;color:#8b949e;margin-top:3px">{noise_pct}% of total</div>'
        f'</div>'
        f'<div style="background:#161b22;border:1px solid rgba(255,107,26,0.2);padding:14px 16px">'
        f'<div style="font-size:9px;color:#8b949e;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:6px">Signal</div>'
        f'<div style="font-size:30px;font-weight:700;color:#00c9b1;line-height:1">{signal_count}</div>'
        f'<div style="font-size:9px;color:#8b949e;margin-top:3px">{signal_pct}% of total</div>'
        f'</div>'
        f'</div>'
        f'<div style="font-size:9px;color:#ff6b1a;letter-spacing:0.15em;'
        f'text-transform:uppercase;margin-bottom:10px">BADGE BREAKDOWN</div>'
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:5px">'
        f'{cards_html}'
        f'</div>'
        f'</div>'
    )


def handle_inspect():
    """Legacy handler — kept for compatibility."""
    if not state["conversations"]:
        return gr.update(value=""), gr.update(visible=False), gr.update(value=[]), gr.update(value="")
    return (
        gr.update(value=_badge_dashboard_html()),
        gr.update(visible=True),
        gr.update(value=_to_df_rows()),
        gr.update(value=_selection_count_text()),
    )


# ── Browser: sort handlers ─────────────────────────────────────────────────────

def _refresh_browser():
    """Recompute visible_ids and return (df update, count update)."""
    state["visible_ids"] = _apply_sort_filter()
    return gr.update(value=_to_df_rows()), gr.update(value=_selection_count_text())


def handle_sort_date():
    state["sort_by"] = "date"
    return _refresh_browser()


def handle_sort_size():
    state["sort_by"] = "size"
    return _refresh_browser()


def handle_sort_title():
    state["sort_by"] = "title"
    return _refresh_browser()


# ── Browser: filter handler ────────────────────────────────────────────────────

def handle_apply_filter(from_date, to_date):
    state["from_date"] = from_date.strip()
    state["to_date"] = to_date.strip()
    return _refresh_browser()


def handle_clear_filter():
    state["from_date"] = ""
    state["to_date"] = ""
    return (
        gr.update(value=""),
        gr.update(value=""),
        *_refresh_browser(),
    )


# ── Browser: select / deselect ────────────────────────────────────────────────

def handle_select_all():
    for id_ in state["visible_ids"]:
        state["conv_table"][id_]["selected"] = True
    return gr.update(value=_to_df_rows()), gr.update(value=_selection_count_text())


def handle_deselect_all():
    for id_ in state["visible_ids"]:
        state["conv_table"][id_]["selected"] = False
    return gr.update(value=_to_df_rows()), gr.update(value=_selection_count_text())


def handle_deselect_noise():
    for id_ in state["visible_ids"]:
        if state["conv_table"][id_].get("noise", False):
            state["conv_table"][id_]["selected"] = False
    return gr.update(value=_to_df_rows()), gr.update(value=_selection_count_text())


# ── Browser: sync checkbox edits back to state ────────────────────────────────

def handle_table_change(df_data):
    if df_data is None or not state["visible_ids"]:
        return gr.update(value=_selection_count_text())

    if hasattr(df_data, "values"):
        rows = df_data.values.tolist()
    else:
        rows = list(df_data)

    for i, row in enumerate(rows):
        if i < len(state["visible_ids"]):
            id_ = state["visible_ids"][i]
            if isinstance(row, (list, tuple)) and len(row) > 0:
                state["conv_table"][id_]["selected"] = bool(row[0])

    return gr.update(value=_selection_count_text())


# ── Step 2: Analyze ───────────────────────────────────────────────────────────

def _progress_html(text: str, current: int = 0, total: int = 0) -> str:
    if total > 0:
        pct = int(current / total * 100)
        bar = (
            f'<div style="display:flex;align-items:center;gap:8px;margin-top:6px">'
            f'<div style="flex:1;background:#e0e0e0;border-radius:4px;height:10px">'
            f'<div style="width:{pct}%;background:#4a90d9;border-radius:4px;height:10px;transition:width 0.2s"></div>'
            f'</div>'
            f'<span style="font-size:0.85em;min-width:38px;text-align:right;color:#555">{pct}%</span>'
            f'</div>'
        )
    else:
        bar = ""
    return f'<div style="font-family:sans-serif;padding:4px 0">{text}{bar}</div>'


def handle_analyze():
    def _yields(status, output, choices=None):
        update = gr.update(choices=choices, value=None) if choices is not None else gr.update()
        yield status, output, update, gr.update(), gr.update()

    if not state["conversations"]:
        yield from _yields(
            "",
            "❌ No conversations loaded. Please upload your export first.",
            [],
        )
        return

    if state["analysis_complete"]:
        yield from _yields(
            "",
            "⚠️ Analysis has already been run on this export.\n\n"
            "If you want to re-analyze, please upload your export again on the "
            "**Upload** tab — this ensures your previous results are not lost.\n\n"
            "If you want to continue to export, click the **Export** tab.",
        )
        return

    if not check_ollama_running():
        yield from _yields(
            "",
            "❌ Ollama is not running.\n\n"
            "Please start Ollama and try again.\n"
            "On Windows: Open the Ollama app from your Start menu.\n"
            "Then run: `ollama pull mistral-nemo`",
            [],
        )
        return

    conversations = get_selected_conversations(state["conversations"], state["conv_table"])
    conversations = apply_test_mode(conversations)

    ollama_cfg = config.get_ollama_config()
    total = len(conversations)
    summaries = []
    start_time = time.time()

    def _elapsed_str():
        s = int(time.time() - start_time)
        return f"{s // 60}:{s % 60:02d}"

    yield (
        _progress_html("⏳ Loading model into memory — first summary may take 20–30 seconds…"),
        "",
        gr.update(),
        gr.update(),
        gr.update(),
    )

    summary_style = config.get("summary_style", "concise")
    state["ollama_model"] = ollama_cfg["model"]
    for current, total, title, summary in stage4_extract_gen(
        conversations,
        model=ollama_cfg["model"],
        summary_style=summary_style,
    ):
        if summary:
            summaries.append(summary)
        status_text = f"⏳ Summarizing {current} of {total}: {title} — elapsed: {_elapsed_str()}"
        yield _progress_html(status_text, current, total), "", gr.update(), gr.update(), gr.update()

    state["summaries"] = summaries
    classify_result = stage5_classify(summaries)
    state["grouped"] = classify_result["grouped"]
    state["analysis_complete"] = True

    stats = classify_result["stats"]
    bucket_names = classify_result["bucket_names"]

    yield (
        _progress_html(f"⏳ Synthesizing biography profile — elapsed: {_elapsed_str()}"),
        "",
        gr.update(choices=bucket_names, value=None),
        gr.update(),
        gr.update(),
    )

    model = state.get("ollama_model") or config.get_ollama_config()["model"]
    out_dir = config.get_output_dir()
    synth = stage6_synthesize(state["summaries"], out_dir, model=model)
    result = None if "error" in synth else synth
    state["export_dir"] = str(out_dir)
    if result and "psychographic" in result:
        state["psychographic"] = result["psychographic"]

    lines = [f"✅ Analysis complete — **{len(summaries)}/{total} summarized in {_elapsed_str()}**\n"]
    if result is None:
        lines.append(
            "⚠️ Biography synthesis failed — try running Export to retry, "
            "or switch to a larger model in Settings.\n"
        )
    lines.append("### Detected Topics:\n")
    for s in stats:
        count = s['count']
        label = "conversation" if count == 1 else "conversations"
        lines.append(f"- **{s['bucket']}**: {count} {label}")

    yield (
        "",
        "\n".join(lines),
        gr.update(choices=bucket_names, value=None),
        _read_biography(),
        _render_psychographic_html(state["psychographic"]),
    )


# ── Step 3: Review ────────────────────────────────────────────────────────────

def _tier_badge(tier: str) -> str:
    colours = {"signal": "#1a7f4b", "noise": "#888", "review": "#b45309"}
    bg = colours.get(tier, "#555")
    return f'<span style="background:{bg};color:#fff;padding:2px 8px;border-radius:10px;font-size:0.8em;font-weight:600">{tier or "?"}</span>'


def _pill_list(items: list) -> str:
    if not items:
        return ""
    pills = "".join(
        f'<span style="background:#1a1a2e;border:1px solid rgba(255,107,26,0.2);color:#e6edf3;padding:1px 7px;border-radius:10px;font-size:0.82em;margin:2px 2px 2px 0;display:inline-block">{i}</span>'
        for i in items
    )
    return pills


def _section(label: str, html_body: str) -> str:
    return f'<div style="margin-top:8px"><span style="font-weight:600;color:#ff6b1a">{label}</span> <span style="color:#e6edf3">{html_body}</span></div>'


def handle_view_summaries(bucket_name: str) -> str:
    if not bucket_name or not state["grouped"]:
        return ""
    convos = state["grouped"].get(bucket_name, [])
    if not convos:
        return f"<em>No conversations in <strong>{bucket_name}</strong>.</em>"

    parts = []
    for c in convos:
        title = c.get("title", "Untitled")
        date = c.get("created", "")
        tier = c.get("tier", "")
        topic = c.get("topic", "")

        who = c.get("who") or {}
        if not isinstance(who, dict):
            who = {}
        what = c.get("what") or {}
        if not isinstance(what, dict):
            what = {}
        how = c.get("how") or {}
        if not isinstance(how, dict):
            how = {}
        names = c.get("names") or []
        if not isinstance(names, list):
            names = []

        rows = []

        who_parts = []
        if who.get("role"):
            who_parts.append(f"<em>Role:</em> {who['role']}")
        if who.get("expertise"):
            who_parts.append(f"<em>Expertise:</em> {_pill_list(who['expertise'])}")
        if who.get("interests"):
            who_parts.append(f"<em>Interests:</em> {_pill_list(who['interests'])}")
        if who.get("help_seeking"):
            who_parts.append(f"<em>Help sought:</em> {_pill_list(who['help_seeking'])}")
        if who.get("credentials"):
            who_parts.append(f"<em>Credentials:</em> {_pill_list(who['credentials'])}")
        if who_parts:
            rows.append(_section("WHO", " &nbsp;·&nbsp; ".join(who_parts)))

        what_parts = []
        if what.get("goal"):
            what_parts.append(what["goal"])
        if what.get("project"):
            what_parts.append(f"<em>Project:</em> {what['project']}")
        outcome = what.get("outcome")
        if outcome:
            what_parts.append(f"<em>Outcome:</em> {outcome}")
        if what.get("open_thread"):
            what_parts.append(f"<em>Open:</em> {what['open_thread']}")
        if what_parts:
            rows.append(_section("WHAT", " &nbsp;·&nbsp; ".join(what_parts)))

        how_parts = []
        if how.get("tone"):
            how_parts.append(f"<em>Tone:</em> {how['tone']}")
        if how.get("corrections"):
            how_parts.append(f"<em>Corrections:</em> {_pill_list(how['corrections'])}")
        if how.get("your_words"):
            quoted = " ".join(f'"{w}"' for w in how["your_words"])
            how_parts.append(f"<em>Your words:</em> {quoted}")
        if how_parts:
            rows.append(_section("HOW", " &nbsp;·&nbsp; ".join(how_parts)))

        if names:
            rows.append(_section("Names", _pill_list(names)))

        raw_json = json.dumps(c, indent=2, ensure_ascii=False)
        raw_toggle = (
            f'<details style="margin-top:10px">'
            f'<summary style="cursor:pointer;color:#8b949e;font-size:0.82em">Raw JSON</summary>'
            f'<pre style="background:#0d1117;border:1px solid rgba(255,107,26,0.2);color:#8b949e;'
            f'padding:10px;border-radius:4px;font-family:monospace;'
            f'font-size:0.78em;overflow-x:auto;white-space:pre-wrap">{raw_json}</pre>'
            f'</details>'
        )

        body = "".join(rows) + raw_toggle
        card = (
            f'<div style="background:#161b22;border:1px solid rgba(255,107,26,0.2);border-radius:8px;padding:14px 16px;margin-bottom:14px">'
            f'<div style="display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;margin-bottom:4px">'
            f'<span style="font-size:1.05em;font-weight:700;color:#e6edf3">{title}</span>'
            f'{_tier_badge(tier)}'
            f'<span style="color:#8b949e;font-size:0.82em">{topic}</span>'
            f'</div>'
            f'<div style="color:#8b949e;font-size:0.82em;margin-bottom:8px">{date}</div>'
            f'{body}'
            f'</div>'
        )
        parts.append(card)

    return "".join(parts)


# ── Step 4: Export ────────────────────────────────────────────────────────────

def _read_biography() -> str:
    bio_path = config.get_output_dir() / "get_to_know_me.md"
    if not bio_path.exists():
        return "_Run Export to generate your biography profile._"
    try:
        return bio_path.read_text(encoding="utf-8")
    except OSError:
        return "_Run Export to generate your biography profile._"


def _render_psychographic_html(psychographic: dict) -> str:
    if not psychographic or not psychographic.get("axes"):
        return ""

    parts = []
    for ax in psychographic.get("axes", []):
        name = ax.get("axis", "")
        try:
            score = float(ax.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        archetype = ax.get("archetype", "")
        evidence = ax.get("evidence", "")
        pct = (score / 5.0) * 100

        bar = (
            f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0">'
            f'<div style="width:140px;background:#1a1a2e;border-radius:3px;height:8px;flex-shrink:0">'
            f'<div style="width:{pct:.0f}%;background:#ff6b1a;border-radius:3px;height:8px"></div>'
            f'</div>'
            f'<span style="color:#e6edf3;font-size:0.85em">{score:.1f} / 5.0</span>'
            f'</div>'
        )
        parts.append(
            f'<div style="margin-bottom:14px">'
            f'<div><span style="color:#ff6b1a;font-weight:600">{name}</span>'
            f'<span style="color:#e6edf3;font-size:0.9em"> — {archetype}</span></div>'
            f'{bar}'
            f'<div style="color:#8b949e;font-size:0.82em;font-style:italic">{evidence}</div>'
            f'</div>'
        )

    composite = psychographic.get("composite_archetype", "")
    archetype_summary = psychographic.get("archetype_summary", "")
    footer = ""
    if composite:
        footer += f'<div style="margin-top:16px;font-size:1.2em;font-weight:700;color:#ff6b1a">{composite}</div>'
    if archetype_summary:
        footer += f'<div style="color:#e6edf3;margin-top:6px">{archetype_summary}</div>'

    return (
        f'<div style="background:#0d1117;border:1px solid rgba(255,107,26,0.2);'
        f'border-radius:8px;padding:16px 20px;margin-bottom:16px">'
        + "".join(parts) + footer
        + f'</div>'
    )


def handle_export(output_path: str):
    _no_change = gr.update()

    if not state["grouped"] or not state["summaries"]:
        return "❌ No data to export. Please analyze your conversations first.", _no_change

    if not output_path.strip():
        output_path = str(config.get_output_dir())

    if len(output_path) > 200:
        return "❌ Output path is too long. Please choose a shorter folder path.", _no_change

    try:
        out_dir = Path(output_path)
        export_result = stage6_export(state["grouped"], state["summaries"], out_dir)
        if "error" in export_result:
            raise Exception(export_result["error"])
        exported = export_result["exported"]

        src_bio = config.get_output_dir() / "get_to_know_me.md"
        dst_bio = out_dir / "get_to_know_me.md"
        if src_bio.exists() and src_bio.resolve() != dst_bio.resolve():
            try:
                dst_bio.write_text(src_bio.read_text(encoding="utf-8"), encoding="utf-8")
            except OSError as e:
                print(f"[warn] Could not copy get_to_know_me.md to export dir: {e}")

        lines = [f"✅ Export complete — **{len(exported)} files written**\n"]
        lines.append(f"📁 Output folder: `{out_dir}`\n")
        lines.append("### Files created:\n")
        lines.append(f"- 📖 `get_to_know_me.md` ← Biography profile synthesized from all conversations")
        for bucket, path in exported.items():
            if bucket == "__master__":
                lines.append(f"- 📋 `{path.name}` ← Upload this to your new LLM first")
            else:
                lines.append(f"- `{path.name}`")

        lines.append("\n---")
        lines.append("### Next Steps:")
        lines.append("1. Go to [claude.ai](https://claude.ai) → Projects → New Project")
        lines.append("2. Upload the context files for each topic")
        lines.append("3. Start chatting — your context is restored!")

        config.set_value("last_export_path", str(out_dir))
        state["export_dir"] = str(out_dir)
        return "\n".join(lines), _read_biography()

    except PermissionError:
        return "❌ Permission denied. Please choose a folder you have write access to.", _no_change
    except OSError as e:
        return f"❌ Could not write to folder: {e}", _no_change
    except Exception as e:
        return f"❌ Export failed: {e}", _no_change


# ── Dark theme CSS ────────────────────────────────────────────────────────────

DARK_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Barlow+Condensed:wght@300;400;600;800&family=Barlow:wght@300;400;500&display=swap');

body, .gradio-container, .main, footer {
    background: #080c10 !important;
    color: #e6edf3 !important;
    font-family: 'Barlow', sans-serif !important;
}

body::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(255,107,26,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,107,26,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

.tab-nav {
    border-bottom: 1px solid rgba(255,107,26,0.2) !important;
    background: #0d1117 !important;
}
.tab-nav button {
    background: #0d1117 !important;
    color: #8b949e !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 12px 20px !important;
    border-radius: 0 !important;
    transition: all 0.2s !important;
}
.tab-nav button:hover {
    color: #ff6b1a !important;
    background: #161b22 !important;
}
.tab-nav button.selected {
    background: #080c10 !important;
    color: #ff6b1a !important;
    border-bottom: 2px solid #ff6b1a !important;
}

input, textarea, select,
.gr-input, .gr-textbox textarea, .gr-textbox input {
    background: #0d1117 !important;
    color: #e6edf3 !important;
    border: 1px solid rgba(255,107,26,0.2) !important;
    border-radius: 0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
}
input:focus, textarea:focus {
    border-color: #ff6b1a !important;
    box-shadow: 0 0 0 2px rgba(255,107,26,0.1) !important;
    outline: none !important;
}
input::placeholder, textarea::placeholder { color: #8b949e !important; }

label span, .gr-label, .block label > span {
    color: #8b949e !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

.prose, .prose p, .prose li, .prose h1, .prose h2, .prose h3,
.gr-markdown, .gr-markdown p {
    color: #e6edf3 !important;
    font-family: 'Barlow', sans-serif !important;
}
.prose h1, .prose h2, .prose h3,
.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    color: #e6edf3 !important;
    letter-spacing: 0.02em !important;
}
.prose strong, .gr-markdown strong { color: #ff6b1a !important; }
.prose code, .gr-markdown code {
    background: #161b22 !important;
    color: #00c9b1 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    padding: 2px 6px !important;
    border-radius: 0 !important;
}

.gr-panel, .gr-box, .block, .contain {
    background: #0d1117 !important;
    border: 1px solid rgba(255,107,26,0.15) !important;
    border-radius: 0 !important;
}

button.primary, .gr-button-primary, button[variant="primary"] {
    background: #ff6b1a !important;
    color: #000 !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    transition: all 0.2s !important;
}
button.primary:hover, .gr-button-primary:hover {
    background: #fff !important;
    color: #000 !important;
    transform: translateY(-1px) !important;
}

button.secondary, .gr-button-secondary, button[variant="secondary"] {
    background: #161b22 !important;
    color: #e6edf3 !important;
    border: 1px solid rgba(255,107,26,0.2) !important;
    border-radius: 0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    transition: all 0.2s !important;
}
button.secondary:hover {
    border-color: #ff6b1a !important;
    color: #ff6b1a !important;
}

.gr-dropdown ul, ul.options {
    background: #161b22 !important;
    border: 1px solid rgba(255,107,26,0.2) !important;
    border-radius: 0 !important;
}
.gr-dropdown ul li:hover { background: #1e2530 !important; color: #ff6b1a !important; }

.gr-dataframe table {
    background: #0d1117 !important;
    color: #e6edf3 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
}
.gr-dataframe th {
    background: #161b22 !important;
    color: #ff6b1a !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-color: rgba(255,107,26,0.2) !important;
}
.gr-dataframe td { border-color: rgba(255,107,26,0.08) !important; }
.gr-dataframe tr:hover td { background: rgba(255,107,26,0.04) !important; }

.gr-html, .output-html {
    background: #0d1117 !important;
    color: #e6edf3 !important;
    font-family: 'Space Mono', monospace !important;
    border: 1px solid rgba(255,107,26,0.15) !important;
    border-top: 3px solid #ff6b1a !important;
    border-radius: 0 !important;
    padding: 16px !important;
}

.progress-bar { background: #ff6b1a !important; }
.progress-bar-wrap { background: #161b22 !important; border-radius: 0 !important; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: rgba(255,107,26,0.4); border-radius: 0; }
::-webkit-scrollbar-thumb:hover { background: #ff6b1a; }

.upload-container, [data-testid="file-upload"] {
    background: #0d1117 !important;
    border: 1px dashed rgba(255,107,26,0.3) !important;
    border-radius: 0 !important;
    transition: border-color 0.2s !important;
}
.upload-container:hover, [data-testid="file-upload"]:hover {
    border-color: #ff6b1a !important;
}

.gr-accordion { border-color: rgba(255,107,26,0.2) !important; border-radius: 0 !important; }
.gr-accordion-header {
    background: #161b22 !important;
    color: #e6edf3 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
}
"""


# ── Step indicator ────────────────────────────────────────────────────────────

_STEP_LABELS = ["Setup", "Upload", "Inspect", "Extract", "Review", "Export"]


def step_indicator_html(current: int) -> str:
    parts = []
    for i, label in enumerate(_STEP_LABELS):
        if i < current:
            dot_bg = "#00c9b1"
            dot_border = "#00c9b1"
            label_color = "#00c9b1"
        elif i == current:
            dot_bg = "#ff6b1a"
            dot_border = "#ff6b1a"
            label_color = "#ff6b1a"
        else:
            dot_bg = "#1a1a2e"
            dot_border = "rgba(255,107,26,0.2)"
            label_color = "#8b949e"

        parts.append(
            f'<div style="display:flex;flex-direction:column;align-items:center;gap:5px">'
            f'<div style="width:14px;height:14px;border-radius:50%;'
            f'background:{dot_bg};border:2px solid {dot_border}"></div>'
            f'<span style="font-family:\'Space Mono\',monospace;font-size:9px;'
            f'color:{label_color};letter-spacing:0.06em;text-transform:uppercase">'
            f'{label}</span>'
            f'</div>'
        )
        if i < len(_STEP_LABELS) - 1:
            connector_bg = "#00c9b1" if i < current else "rgba(255,107,26,0.15)"
            parts.append(
                f'<div style="flex:1;height:2px;background:{connector_bg};'
                f'margin-bottom:20px;min-width:24px"></div>'
            )

    return (
        f'<div style="padding:14px 32px;background:#080c10;'
        f'border-bottom:1px solid rgba(255,107,26,0.15)">'
        f'<div style="display:flex;align-items:center;max-width:640px">'
        + "".join(parts)
        + f'</div></div>'
    )


# ── UI Layout ─────────────────────────────────────────────────────────────────

def build_ui():
    cfg = config.load_config()

    with gr.Blocks(
        title="OwnYourContext — LLM Conversation Memory Migrator",
    ) as app:

        # ── Header ──────────────────────────────────────────────────────────
        gr.HTML("""
            <div style="border-bottom:1px solid rgba(255,107,26,0.2); padding:24px 32px 22px; background:#080c10;">
                <div style="font-family:'Space Mono',monospace; font-size:10px; color:#ff6b1a; letter-spacing:0.2em; text-transform:uppercase; margin-bottom:12px;">
                    // V0.3 &middot; OPEN SOURCE &middot; 100% LOCAL &middot; NO CLOUD
                </div>
                <div style="font-family:'Barlow Condensed',sans-serif; font-weight:800; font-size:38px; text-transform:uppercase; letter-spacing:0.02em; color:#e6edf3; line-height:1; margin-bottom:6px;">
                    OWNYOURCONTEXT
                </div>
                <div style="font-family:'Barlow Condensed',sans-serif; font-weight:300; font-size:15px; color:#8b949e; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:18px;">
                    YOUR AI MEMORY. YOUR MACHINE. YOUR CALL.
                </div>
                <div style="display:flex; gap:8px; flex-wrap:wrap;">
                    <span style="display:inline-flex; align-items:center; gap:6px; background:rgba(0,201,177,0.05); border:1px solid rgba(0,201,177,0.25); color:#00c9b1; font-family:'Space Mono',monospace; font-size:10px; letter-spacing:0.08em; padding:5px 12px;">🔒 100% Local</span>
                    <span style="display:inline-flex; align-items:center; gap:6px; background:rgba(0,201,177,0.05); border:1px solid rgba(0,201,177,0.25); color:#00c9b1; font-family:'Space Mono',monospace; font-size:10px; letter-spacing:0.08em; padding:5px 12px;">🚫 No Cloud</span>
                    <span style="display:inline-flex; align-items:center; gap:6px; background:rgba(0,201,177,0.05); border:1px solid rgba(0,201,177,0.25); color:#00c9b1; font-family:'Space Mono',monospace; font-size:10px; letter-spacing:0.08em; padding:5px 12px;">🛡️ No Data Sharing</span>
                    <span style="display:inline-flex; align-items:center; gap:6px; background:rgba(0,201,177,0.05); border:1px solid rgba(0,201,177,0.25); color:#00c9b1; font-family:'Space Mono',monospace; font-size:10px; letter-spacing:0.08em; padding:5px 12px;">✅ Open Source — MIT</span>
                </div>
            </div>
        """)

        # ── Step indicator ───────────────────────────────────────────────────
        step_indicator = gr.HTML(value=step_indicator_html(0))

        # ── Wizard helpers ───────────────────────────────────────────────────
        def show_step(n):
            return [gr.update(visible=(i == n)) for i in range(6)]

        def _nav(n):
            return [gr.update(value=step_indicator_html(n))] + show_step(n)

        # ── Step 0: Setup ────────────────────────────────────────────────────
        with gr.Column(visible=True) as col_step0:
            gr.Markdown("## Step 0 — Setup")
            gr.Markdown("Step 0 content coming soon")
            with gr.Row():
                back0 = gr.Button("← Back", visible=False)
                next0 = gr.Button("Next →", variant="primary")

        # ── Step 1: Upload ───────────────────────────────────────────────────
        with gr.Column(visible=True) as col_step1:
            gr.Markdown(
                "## Step 1 — Upload\n\n"
                "Export your conversations from ChatGPT: **Settings → Data Controls → Export data**.\n\n"
                "Once the download email arrives, locate the `.zip` file and drop it below."
            )
            zip_upload = gr.File(
                label="ChatGPT Export ZIP",
                file_types=[".zip"],
                type="binary",
            )
            load_btn = gr.Button("LOAD EXPORT", variant="primary")
            upload_msg = gr.Markdown("")
            inspect_btn = gr.Button("INSPECT →", variant="primary", visible=False)
            with gr.Row():
                back1 = gr.Button("← Back")

        # ── Step 2: Inspect ──────────────────────────────────────────────────
        with gr.Column(visible=True) as col_step2:
            gr.Markdown("## Step 2 — Inspect\n\nReview your conversations before running analysis.")
            scan_btn = gr.Button("RUN BADGE SCAN", variant="primary")
            badge_dashboard_html = gr.HTML("")
            selection_count = gr.Markdown("")
            with gr.Row():
                sort_date_btn = gr.Button("Date", variant="secondary")
                sort_size_btn = gr.Button("Size", variant="secondary")
                sort_title_btn = gr.Button("Title", variant="secondary")
            with gr.Row():
                from_date_tb = gr.Textbox(label="From (YYYY-MM-DD)", placeholder="2024-01-01", scale=2)
                to_date_tb = gr.Textbox(label="To (YYYY-MM-DD)", placeholder="2024-12-31", scale=2)
                apply_filter_btn = gr.Button("Apply Filter", variant="secondary")
                clear_filter_btn = gr.Button("Clear", variant="secondary")
            with gr.Row():
                select_all_btn = gr.Button("Select All", variant="secondary")
                deselect_all_btn = gr.Button("Deselect All", variant="secondary")
                deselect_noise_btn = gr.Button("🗑 Deselect Noise", variant="secondary")
            conv_df = gr.Dataframe(
                headers=["✓", "Title", "Date", "Words", "Badges", "Noise"],
                datatype=["bool", "str", "str", "number", "str", "str"],
                interactive=True,
            )
            analyze_btn = gr.Button("ANALYZE SELECTED →", variant="primary")
            with gr.Row():
                back2 = gr.Button("← Back")

        # ── Step 3: Extract ──────────────────────────────────────────────────
        with gr.Column(visible=True) as col_step3:
            gr.Markdown("## Step 3 — Extract")
            gr.Markdown("Step 3 content coming soon")
            with gr.Row():
                back3 = gr.Button("← Back")
                next3 = gr.Button("Next →", variant="primary")

        # ── Step 4: Review ───────────────────────────────────────────────────
        with gr.Column(visible=True) as col_step4:
            gr.Markdown("## Step 4 — Review")
            gr.Markdown("Step 4 content coming soon")
            with gr.Row():
                back4 = gr.Button("← Back")
                next4 = gr.Button("Next →", variant="primary")

        # ── Step 5: Export ───────────────────────────────────────────────────
        with gr.Column(visible=True) as col_step5:
            gr.Markdown("## Step 5 — Export")
            gr.Markdown("Step 5 content coming soon")
            with gr.Row():
                back5 = gr.Button("← Back")
                next5 = gr.Button("Next →", variant="primary", interactive=False)

        # ── Navigation wiring ────────────────────────────────────────────────
        _nav_outputs = [step_indicator, col_step0, col_step1, col_step2, col_step3, col_step4, col_step5]

        next0.click(fn=lambda: _nav(1), outputs=_nav_outputs)
        next3.click(fn=lambda: _nav(4), outputs=_nav_outputs)
        next4.click(fn=lambda: _nav(5), outputs=_nav_outputs)

        back1.click(fn=lambda: _nav(0), outputs=_nav_outputs)
        back2.click(fn=lambda: _nav(1), outputs=_nav_outputs)
        back3.click(fn=lambda: _nav(2), outputs=_nav_outputs)
        back4.click(fn=lambda: _nav(3), outputs=_nav_outputs)
        back5.click(fn=lambda: _nav(4), outputs=_nav_outputs)

        # ── Step 1 wiring ────────────────────────────────────────────────────
        load_btn.click(
            fn=handle_upload,
            inputs=[zip_upload],
            outputs=[upload_msg, inspect_btn],
        )

        # INSPECT → just advances the wizard, no data loading
        inspect_btn.click(
            fn=lambda: _nav(2),
            outputs=_nav_outputs,
        )

        # ── Step 2 wiring ────────────────────────────────────────────────────

        # RUN BADGE SCAN fires independently — raw values only, no nav conflict
        def _run_scan():
            if not state["conversations"]:
                return "", [], ""
            return _badge_dashboard_html(), _to_df_rows(), _selection_count_text()

        scan_btn.click(
            fn=_run_scan,
            outputs=[badge_dashboard_html, conv_df, selection_count],
        )

        sort_date_btn.click(fn=handle_sort_date, outputs=[conv_df, selection_count])
        sort_size_btn.click(fn=handle_sort_size, outputs=[conv_df, selection_count])
        sort_title_btn.click(fn=handle_sort_title, outputs=[conv_df, selection_count])

        apply_filter_btn.click(
            fn=handle_apply_filter,
            inputs=[from_date_tb, to_date_tb],
            outputs=[conv_df, selection_count],
        )
        clear_filter_btn.click(
            fn=handle_clear_filter,
            outputs=[from_date_tb, to_date_tb, conv_df, selection_count],
        )

        select_all_btn.click(fn=handle_select_all, outputs=[conv_df, selection_count])
        deselect_all_btn.click(fn=handle_deselect_all, outputs=[conv_df, selection_count])
        deselect_noise_btn.click(fn=handle_deselect_noise, outputs=[conv_df, selection_count])

        conv_df.change(fn=handle_table_change, inputs=[conv_df], outputs=[selection_count])

        analyze_btn.click(fn=lambda: _nav(3), outputs=_nav_outputs)

        app.load(fn=lambda: _nav(0), outputs=_nav_outputs)

    return app


# ── Launch ────────────────────────────────────────────────────────────────────
def _run_scan():
    print(f"[scan] conversations in state: {len(state['conversations'])}")
    if not state["conversations"]:
        return "", [], ""
    return _badge_dashboard_html(), _to_df_rows(), _selection_count_text()
if __name__ == "__main__":
    print("=== LLM Conversation Memory Migrator ===")
    print("Starting local server — no internet connection used.")
    print()

    if not check_ollama_running():
        print("⚠️  Warning: Ollama does not appear to be running.")
        print("   Please start Ollama before analyzing conversations.")
        print("   On Windows: Open the Ollama app from your Start menu.")
        print()

    app = build_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        css=DARK_CSS,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.Color(
                c50="#fff3ee", c100="#ffe0cc", c200="#ffbf99", c300="#ff9d66",
                c400="#ff7c33", c500="#ff6b1a", c600="#cc5200", c700="#993d00",
                c800="#662900", c900="#331400", c950="#1a0a00",
            ),
            secondary_hue=gr.themes.colors.Color(
                c50="#e0faf7", c100="#b3f4ed", c200="#66e9db", c300="#33dfd0",
                c400="#00d4c0", c500="#00c9b1", c600="#00a18e", c700="#00796b",
                c800="#005048", c900="#002824", c950="#001412",
            ),
            neutral_hue=gr.themes.colors.gray,
            font=[gr.themes.GoogleFont("Barlow"), "sans-serif"],
            font_mono=[gr.themes.GoogleFont("Space Mono"), "monospace"],
        ).set(
            body_background_fill="#080c10",
            body_text_color="#e6edf3",
            block_background_fill="#0d1117",
            block_border_color="rgba(255,107,26,0.2)",
            block_border_width="1px",
            block_label_text_color="#8b949e",
            input_background_fill="#0d1117",
            input_border_color="rgba(255,107,26,0.2)",
            button_primary_background_fill="#ff6b1a",
            button_primary_text_color="#000000",
            button_secondary_background_fill="#161b22",
            button_secondary_text_color="#e6edf3",
            button_secondary_border_color="rgba(255,107,26,0.2)",
        ),
    )