"""
app.py
Main entry point for llm-conversation-memory-migrator.
Launches a local Gradio UI — no internet connection required.
"""

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

from adapters.chatgpt import load_from_zip
from core.summarizer import summarize_all, summarize_all_gen, check_ollama_running
from core.classifier import classify_summaries, get_bucket_stats, merge_buckets, rename_bucket, get_all_bucket_names
from core.exporter import export_all
from core import config

# ── State ─────────────────────────────────────────────────────────────────────

state = {
    "conversations": [],
    "summaries": [],
    "grouped": {},
    "export_dir": None,
    "analysis_complete": False,
    # Conversation browser state
    "conv_table": {},    # id -> {id, title, date_str, date_dt, word_count, selected}
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
        table[conv["id"]] = {
            "id": conv["id"],
            "title": conv["title"],
            "date_str": conv["created"],
            "date_dt": _parse_date_str(conv["created"]),
            "word_count": _word_count(conv),
            "selected": True,
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
        rows.append([r["selected"], r["title"], r["date_str"], r["word_count"]])
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
    """Returns (upload_msg, browser_group_update, df_data, selection_count)."""
    _hidden = gr.update(visible=False)
    _empty_df = gr.update(value=[])
    _no_count = gr.update(value="")

    if zip_file is None:
        return (
            "❌ No file uploaded. Please upload your ChatGPT export zip.",
            _hidden, _empty_df, _no_count,
        )
    try:
        tmp_path = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".zip")
            os.close(tmp_fd)
            with open(tmp_path, 'wb') as f:
                f.write(zip_file)
            conversations = load_from_zip(Path(tmp_path))
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        state["conversations"] = conversations
        state["summaries"] = []
        state["grouped"] = {}
        state["analysis_complete"] = False

        # Build browser state
        state["conv_table"] = _build_conv_table(conversations)
        state["sort_by"] = "date"
        state["from_date"] = ""
        state["to_date"] = ""
        state["visible_ids"] = _apply_sort_filter()

        msg = (
            f"✅ Successfully loaded **{len(conversations)} conversations**.\n\n"
            f"Review and select conversations below, then go to **Analyze**.\n\n"
            f"⏱ Estimated time: {len(conversations) // 10}–{len(conversations) // 5} minutes."
        )
        return (
            msg,
            gr.update(visible=True),
            gr.update(value=_to_df_rows()),
            gr.update(value=_selection_count_text()),
        )
    except FileNotFoundError as e:
        return (f"❌ Invalid export file: {e}", _hidden, _empty_df, _no_count)
    except Exception as e:
        return (f"❌ Error loading file: {e}", _hidden, _empty_df, _no_count)


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


# ── Browser: sync checkbox edits back to state ────────────────────────────────

def handle_table_change(df_data):
    if df_data is None or not state["visible_ids"]:
        return gr.update(value=_selection_count_text())

    # df_data may be a pandas DataFrame or list-of-lists
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

def handle_analyze(progress=gr.Progress()):
    def _yields(status, output, choices=None):
        """Helper: yield a consistent 4-tuple for all outputs."""
        update = gr.update(choices=choices, value=None) if choices is not None else gr.update()
        yield status, output, update, update

    if not state["conversations"]:
        yield from _yields(
            "",
            "❌ No conversations loaded. Please upload your export first.",
            [],
        )
        return

    # ── Double-run warning ───────────────────────────────────────────────────
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
            "Then run: `ollama pull llama3.2`",
            [],
        )
        return

    # Respect selection — if a subset is selected, only analyze those
    conversations = state["conversations"]
    if state["conv_table"]:
        selected = [
            c for c in conversations
            if state["conv_table"].get(c["id"], {}).get("selected", True)
        ]
        if 0 < len(selected) < len(conversations):
            conversations = selected

    # Apply test mode slice
    if config.get("test_mode", False):
        n = int(config.get("test_mode_n", 10))
        conversations = conversations[:n]

    ollama_cfg = config.get_ollama_config()
    total = len(conversations)
    summaries = []
    start_time = time.time()

    def _elapsed_str():
        s = int(time.time() - start_time)
        return f"{s // 60}:{s % 60:02d}"

    summary_style = config.get("summary_style", "concise")
    for current, total, title, summary in summarize_all_gen(
        conversations,
        model=ollama_cfg["model"],
        summary_style=summary_style,
    ):
        if summary:
            summaries.append(summary)
        progress(current / total)
        status = f"⏳ Summarizing {current} of {total}: {title} — elapsed: {_elapsed_str()}"
        yield status, "", gr.update(), gr.update()

    state["summaries"] = summaries
    grouped = classify_summaries(summaries)
    state["grouped"] = grouped
    state["analysis_complete"] = True

    stats = get_bucket_stats(grouped)
    bucket_names = get_all_bucket_names(grouped)

    lines = [f"✅ Analysis complete — **{len(summaries)}/{total} summarized in {_elapsed_str()}**\n"]
    lines.append("### Detected Topics:\n")
    for s in stats:
        count = s['count']
        label = "conversation" if count == 1 else "conversations"
        lines.append(f"- **{s['bucket']}**: {count} {label}")

    yield (
        "",
        "\n".join(lines),
        gr.update(choices=bucket_names, value=None),
        gr.update(choices=bucket_names, value=None),
    )


# ── Step 3: Review ────────────────────────────────────────────────────────────

def handle_rename(old_name: str, new_name: str) -> tuple:
    if not old_name or not new_name:
        return "❌ Please enter both current and new bucket names.", gr.update(), gr.update()
    if old_name == new_name:
        return "❌ Names are the same.", gr.update(), gr.update()

    state["grouped"] = rename_bucket(state["grouped"], old_name, new_name)
    bucket_names = get_all_bucket_names(state["grouped"])
    return (
        f"✅ Renamed **{old_name}** to **{new_name}**.",
        gr.update(choices=bucket_names, value=None),
        gr.update(choices=bucket_names, value=None),
    )


def handle_refresh() -> tuple:
    bucket_names = get_all_bucket_names(state["grouped"])
    return (
        gr.update(choices=bucket_names, value=None),
        gr.update(choices=bucket_names, value=None),
    )


def handle_view_summaries(bucket_name: str) -> str:
    if not bucket_name or not state["grouped"]:
        return ""
    convos = state["grouped"].get(bucket_name, [])
    if not convos:
        return f"*No conversations in **{bucket_name}**.*"
    parts = []
    for c in convos:
        title = c.get("title", "Untitled")
        date = c.get("created", "")
        summary = c.get("summary", "*(no summary)*")
        parts.append(f"### {title}\n*{date}*\n\n{summary}")
    return "\n\n---\n\n".join(parts)


# ── Step 4: Export ────────────────────────────────────────────────────────────

def handle_export(output_path: str) -> str:
    if not state["grouped"] or not state["summaries"]:
        return "❌ No data to export. Please analyze your conversations first."

    if not output_path.strip():
        output_path = str(config.get_output_dir())

    # Windows path length guard
    if len(output_path) > 200:
        return "❌ Output path is too long. Please choose a shorter folder path."

    try:
        out_dir = Path(output_path)
        user_name = config.get_user_name()
        exported = export_all(state["grouped"], out_dir, user_name)

        lines = [f"✅ Export complete — **{len(exported)} files written**\n"]
        lines.append(f"📁 Output folder: `{out_dir}`\n")
        lines.append("### Files created:\n")
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
        return "\n".join(lines)

    except PermissionError:
        return "❌ Permission denied. Please choose a folder you have write access to."
    except OSError as e:
        return f"❌ Could not write to folder: {e}"
    except Exception as e:
        return f"❌ Export failed: {e}"


# ── UI Layout ─────────────────────────────────────────────────────────────────

def build_ui():
    cfg = config.load_config()

    with gr.Blocks(
        title="LLM Conversation Memory Migrator",
    ) as app:

        # ── Header ──────────────────────────────────────────────────────────
        gr.HTML("""
            <div style="text-align:center; padding:20px;">
                <h1>🔄 LLM Conversation Memory Migrator</h1>
                <p style="font-size:1.1em; color:#666;">
                    Move your mind, not just your messages.
                </p>
                <div>
                    <span style="background:#1a1a2e; color:#00ff88; padding:8px 16px; border-radius:20px; font-size:0.85em; margin:4px; display:inline-block;">🔒 100% Local</span>
                    <span style="background:#1a1a2e; color:#00ff88; padding:8px 16px; border-radius:20px; font-size:0.85em; margin:4px; display:inline-block;">🚫 No Cloud</span>
                    <span style="background:#1a1a2e; color:#00ff88; padding:8px 16px; border-radius:20px; font-size:0.85em; margin:4px; display:inline-block;">🛡️ No Data Sharing</span>
                    <span style="background:#1a1a2e; color:#00ff88; padding:8px 16px; border-radius:20px; font-size:0.85em; margin:4px; display:inline-block;">✅ Open Source</span>
                </div>
            </div>
        """)

        # ── Step 1: Upload ───────────────────────────────────────────────────
        with gr.Tab("① Upload"):
            gr.Markdown("""
            ### Upload Your ChatGPT Export

            **How to export from ChatGPT:**
            1. Go to chatgpt.com → Settings → Data Controls
            2. Click **Export Data**
            3. Wait for the email from OpenAI
            4. Download the zip file and upload it below
            """)

            upload_input = gr.File(
                label="Upload ChatGPT Export (.zip)",
                file_types=[".zip"],
                type="binary"
            )
            upload_btn = gr.Button("📂 Load Export", variant="primary")
            upload_output = gr.Markdown()

            # ── Conversation browser (hidden until a zip is loaded) ──────────
            with gr.Group(visible=False) as conv_browser_group:
                gr.Markdown("---\n### Conversation Browser")
                gr.Markdown(
                    "Check or uncheck conversations to include/exclude them from analysis. "
                    "If nothing is deselected, all conversations are analyzed."
                )

                # Sort controls
                with gr.Row():
                    gr.Markdown("**Sort:**")
                    sort_date_btn  = gr.Button("📅 Date", size="sm", scale=0)
                    sort_size_btn  = gr.Button("📝 Size", size="sm", scale=0)
                    sort_title_btn = gr.Button("🔤 Title", size="sm", scale=0)

                # Date filter controls
                with gr.Row():
                    from_date_input = gr.Textbox(
                        label="From (YYYY-MM-DD)",
                        placeholder="e.g. 2023-01-01",
                        scale=2,
                    )
                    to_date_input = gr.Textbox(
                        label="To (YYYY-MM-DD)",
                        placeholder="e.g. 2024-12-31",
                        scale=2,
                    )
                    filter_btn = gr.Button("🔍 Apply Filter", scale=1)
                    clear_filter_btn = gr.Button("✖ Clear", scale=1)

                # Selection controls + count
                with gr.Row():
                    select_all_btn   = gr.Button("✅ Select All",   size="sm", scale=0)
                    deselect_all_btn = gr.Button("⬜ Deselect All", size="sm", scale=0)
                    selection_count  = gr.Markdown(value="")

                # The table
                conv_df = gr.Dataframe(
                    headers=["✓", "Title", "Date", "Words"],
                    datatype=["bool", "str", "str", "number"],
                    interactive=True,
                    wrap=False,
                )

            # Wire upload button
            upload_btn.click(
                fn=handle_upload,
                inputs=[upload_input],
                outputs=[upload_output, conv_browser_group, conv_df, selection_count],
            )

            # Wire sort buttons
            sort_date_btn.click(
                fn=handle_sort_date, inputs=[], outputs=[conv_df, selection_count]
            )
            sort_size_btn.click(
                fn=handle_sort_size, inputs=[], outputs=[conv_df, selection_count]
            )
            sort_title_btn.click(
                fn=handle_sort_title, inputs=[], outputs=[conv_df, selection_count]
            )

            # Wire filter buttons
            filter_btn.click(
                fn=handle_apply_filter,
                inputs=[from_date_input, to_date_input],
                outputs=[conv_df, selection_count],
            )
            clear_filter_btn.click(
                fn=handle_clear_filter,
                inputs=[],
                outputs=[from_date_input, to_date_input, conv_df, selection_count],
            )

            # Wire select/deselect buttons
            select_all_btn.click(
                fn=handle_select_all, inputs=[], outputs=[conv_df, selection_count]
            )
            deselect_all_btn.click(
                fn=handle_deselect_all, inputs=[], outputs=[conv_df, selection_count]
            )

            # Sync checkbox edits back to state
            conv_df.change(
                fn=handle_table_change,
                inputs=[conv_df],
                outputs=[selection_count],
            )

        # ── Step 2: Analyze ──────────────────────────────────────────────────
        with gr.Tab("② Analyze"):
            gr.Markdown("""
            ### Analyze Your Conversations

            Local AI (Ollama + Llama 3.2) will read and summarize your
            conversations — entirely on your machine.

            **This may take 5–20 minutes depending on your history size.**
            """)

            analyze_btn = gr.Button(
                "🧠 Analyze with Local AI",
                variant="primary"
            )
            analyze_status = gr.Markdown(value="")
            analyze_output = gr.Markdown()

        # ── Step 3: Review ───────────────────────────────────────────────────
        with gr.Tab("③ Review"):
            gr.Markdown("""
            ### Review & Organize Topics

            Rename or merge topic buckets before exporting.
            """)

            gr.Markdown("**Rename Bucket**")
            with gr.Row():
                rename_old = gr.Dropdown(
                    label="Current name",
                    choices=[],
                    interactive=True
                )
                rename_new = gr.Textbox(label="New name")
            rename_btn = gr.Button("✏️ Rename")
            rename_output = gr.Markdown()

            refresh_btn = gr.Button("🔄 Refresh Bucket List")

            gr.Markdown("---")
            gr.Markdown("**View Summaries**")
            summary_bucket_dd = gr.Dropdown(
                label="View summaries for bucket:",
                choices=[],
                interactive=True,
            )
            summary_viewer = gr.Markdown(value="")

            # Wire analyze button now that dropdowns are defined
            analyze_btn.click(
                fn=handle_analyze,
                inputs=[],
                outputs=[analyze_status, analyze_output, rename_old, summary_bucket_dd]
            )

            refresh_btn.click(
                fn=handle_refresh,
                inputs=[],
                outputs=[rename_old, summary_bucket_dd]
            )

            rename_btn.click(
                fn=handle_rename,
                inputs=[rename_old, rename_new],
                outputs=[rename_output, rename_old, summary_bucket_dd]
            )

            summary_bucket_dd.change(
                fn=handle_view_summaries,
                inputs=[summary_bucket_dd],
                outputs=[summary_viewer],
            )

        # ── Step 4: Export ───────────────────────────────────────────────────
        with gr.Tab("④ Export"):
            gr.Markdown("""
            ### Export Your Context Files

            Generates clean markdown files ready to upload to Claude Projects
            (or any other LLM service).
            """)

            output_dir_input = gr.Textbox(
                label="Output folder",
                value=str(config.get_output_dir()),
                placeholder="Leave blank to use default"
            )
            export_btn = gr.Button("💾 Export Context Files", variant="primary")
            export_output = gr.Markdown()

            export_btn.click(
                fn=handle_export,
                inputs=[output_dir_input],
                outputs=[export_output]
            )

        # ── Settings ─────────────────────────────────────────────────────────
        with gr.Tab("⚙️ Settings"):
            gr.Markdown("### Settings")

            user_name_input = gr.Textbox(
                label="Your name (optional — included in master context doc)",
                value=cfg.get("user_name", ""),
                placeholder="e.g. Don"
            )

            ollama_model_input = gr.Dropdown(
                label="Ollama model",
                choices=[
                    ("llama3.2 (default, fastest — 3B)", "llama3.2"),
                    ("llama3.1:8b (better quality — 8B)", "llama3.1:8b"),
                    ("mistral-nemo:12b (best quality — 12B)", "mistral-nemo:12b"),
                    ("mistral:7b (fast alternative — 7B)", "mistral:7b"),
                ],
                value=cfg.get("ollama_model", "llama3.2"),
                allow_custom_value=True,
                info="Larger models produce richer summaries but run slower. All models run 100% locally on your machine via Ollama.",
            )

            ollama_url_input = gr.Textbox(
                label="Ollama URL (advanced)",
                value=cfg.get("ollama_url", "http://localhost:11434"),
            )

            summary_style_input = gr.Dropdown(
                label="Summary Style",
                choices=["Concise (4-6 sentences)", "Detailed (8-20 bullet points)"],
                value="Detailed (8-20 bullet points)" if cfg.get("summary_style") == "detailed" else "Concise (4-6 sentences)",
            )

            gr.Markdown("---\n**Test Mode**")
            test_mode_input = gr.Checkbox(
                label="Test Mode — analyze first N conversations only",
                value=cfg.get("test_mode", False),
            )
            test_mode_n_input = gr.Number(
                label="N",
                value=cfg.get("test_mode_n", 10),
                precision=0,
                minimum=1,
            )

            save_settings_btn = gr.Button("💾 Save Settings")
            settings_output = gr.Markdown()

            def save_settings(name, model, url, style, test_mode, test_mode_n):
                config.set_value("user_name", name)
                config.set_value("ollama_model", model)
                config.set_value("ollama_url", url)
                config.set_value("summary_style", "detailed" if "Detailed" in style else "concise")
                config.set_value("test_mode", bool(test_mode))
                config.set_value("test_mode_n", int(test_mode_n) if test_mode_n else 10)
                return "✅ Settings saved."

            save_settings_btn.click(
                fn=save_settings,
                inputs=[user_name_input, ollama_model_input, ollama_url_input, summary_style_input, test_mode_input, test_mode_n_input],
                outputs=[settings_output]
            )

    return app


# ── Launch ────────────────────────────────────────────────────────────────────

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
        theme=gr.themes.Soft(),
    )
