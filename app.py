"""
app.py
Main entry point for llm-conversation-memory-migrator.
Launches a local Gradio UI — no internet connection required.
"""

import os

# ── Privacy: disable all huggingface and gradio telemetry ────────────────────
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

import gradio as gr
from pathlib import Path

from adapters.chatgpt import load_from_zip
from core.summarizer import summarize_all, check_ollama_running
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
}


# ── Step 1: Upload & Parse ────────────────────────────────────────────────────

def handle_upload(zip_file) -> str:
    if zip_file is None:
        return "❌ No file uploaded. Please upload your ChatGPT export zip."
    try:
        zip_path = Path(zip_file.name)
        conversations = load_from_zip(zip_path)
        state["conversations"] = conversations
        state["summaries"] = []
        state["grouped"] = {}
        state["analysis_complete"] = False
        return (
            f"✅ Successfully loaded **{len(conversations)} conversations**.\n\n"
            f"Click **Analyze** to start summarizing with local AI.\n\n"
            f"⏱ Estimated time: {len(conversations) // 10}–{len(conversations) // 5} minutes."
        )
    except FileNotFoundError as e:
        return f"❌ Invalid export file: {e}"
    except Exception as e:
        return f"❌ Error loading file: {e}"


# ── Step 2: Analyze ───────────────────────────────────────────────────────────

def handle_analyze(progress=gr.Progress(track_tqdm=True)):
    if not state["conversations"]:
        return (
            "❌ No conversations loaded. Please upload your export first.",
            gr.update(choices=[]),
            gr.update(choices=[]),
            gr.update(choices=[]),
        )

    # ── Double-run warning ───────────────────────────────────────────────────
    if state["analysis_complete"]:
        return (
            "⚠️ Analysis has already been run on this export.\n\n"
            "If you want to re-analyze, please upload your export again on the "
            "**Upload** tab — this ensures your previous results are not lost.\n\n"
            "If you want to continue to export, click the **Export** tab.",
            gr.update(),
            gr.update(),
            gr.update(),
        )

    if not check_ollama_running():
        return (
            "❌ Ollama is not running.\n\n"
            "Please start Ollama and try again.\n"
            "On Windows: Open the Ollama app from your Start menu.\n"
            "Then run: `ollama pull llama3.2`",
            gr.update(choices=[]),
            gr.update(choices=[]),
            gr.update(choices=[]),
        )

    ollama_cfg = config.get_ollama_config()
    total = len(state["conversations"])

    def update_progress(current, total):
        progress(current / total, desc=f"Summarizing {current} of {total} conversations...")

    summaries = summarize_all(
        state["conversations"],
        model=ollama_cfg["model"],
        progress_callback=update_progress
    )

    state["summaries"] = summaries
    grouped = classify_summaries(summaries)
    state["grouped"] = grouped
    state["analysis_complete"] = True

    stats = get_bucket_stats(grouped)
    bucket_names = get_all_bucket_names(grouped)

    lines = [f"✅ Analysis complete — **{len(summaries)}/{total} conversations summarized**\n"]
    lines.append("### Detected Topics:\n")
    for s in stats:
        count = s['count']
        label = "conversation" if count == 1 else "conversations"
        lines.append(f"- **{s['bucket']}**: {count} {label}")

    return (
        "\n".join(lines),
        gr.update(choices=bucket_names, value=None),
        gr.update(choices=bucket_names, value=None),
        gr.update(choices=bucket_names, value=None),
    )


# ── Step 3: Review ────────────────────────────────────────────────────────────

def handle_merge(source: str, target: str) -> tuple:
    if not source or not target:
        return "❌ Please select both a source and target bucket.", gr.update(), gr.update(), gr.update()
    if source == target:
        return "❌ Source and target must be different.", gr.update(), gr.update(), gr.update()

    state["grouped"] = merge_buckets(state["grouped"], source, target)
    bucket_names = get_all_bucket_names(state["grouped"])
    return (
        f"✅ Merged **{source}** into **{target}**.",
        gr.update(choices=bucket_names, value=None),
        gr.update(choices=bucket_names, value=None),
        gr.update(choices=bucket_names, value=None),
    )


def handle_rename(old_name: str, new_name: str) -> tuple:
    if not old_name or not new_name:
        return "❌ Please enter both current and new bucket names.", gr.update(), gr.update(), gr.update()
    if old_name == new_name:
        return "❌ Names are the same.", gr.update(), gr.update(), gr.update()

    state["grouped"] = rename_bucket(state["grouped"], old_name, new_name)
    bucket_names = get_all_bucket_names(state["grouped"])
    return (
        f"✅ Renamed **{old_name}** to **{new_name}**.",
        gr.update(choices=bucket_names, value=None),
        gr.update(choices=bucket_names, value=None),
        gr.update(choices=bucket_names, value=None),
    )


def handle_refresh() -> tuple:
    bucket_names = get_all_bucket_names(state["grouped"])
    return (
        gr.update(choices=bucket_names, value=None),
        gr.update(choices=bucket_names, value=None),
        gr.update(choices=bucket_names, value=None),
    )


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
                type="filepath"
            )
            upload_btn = gr.Button("📂 Load Export", variant="primary")
            upload_output = gr.Markdown()

            upload_btn.click(
                fn=handle_upload,
                inputs=[upload_input],
                outputs=[upload_output]
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
            analyze_output = gr.Markdown()

        # ── Step 3: Review ───────────────────────────────────────────────────
        with gr.Tab("③ Review"):
            gr.Markdown("""
            ### Review & Organize Topics

            Rename or merge topic buckets before exporting.
            """)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Merge Buckets**")
                    merge_source = gr.Dropdown(
                        label="Merge from",
                        choices=[],
                        interactive=True
                    )
                    merge_target = gr.Dropdown(
                        label="Merge into",
                        choices=[],
                        interactive=True
                    )
                    merge_btn = gr.Button("🔀 Merge")
                    merge_output = gr.Markdown()

                with gr.Column():
                    gr.Markdown("**Rename Bucket**")
                    rename_old = gr.Dropdown(
                        label="Current name",
                        choices=[],
                        interactive=True
                    )
                    rename_new = gr.Textbox(label="New name")
                    rename_btn = gr.Button("✏️ Rename")
                    rename_output = gr.Markdown()

            refresh_btn = gr.Button("🔄 Refresh Bucket List")

            # Wire analyze button now that dropdowns are defined
            analyze_btn.click(
                fn=handle_analyze,
                inputs=[],
                outputs=[analyze_output, merge_source, merge_target, rename_old]
            )

            refresh_btn.click(
                fn=handle_refresh,
                inputs=[],
                outputs=[merge_source, merge_target, rename_old]
            )

            merge_btn.click(
                fn=handle_merge,
                inputs=[merge_source, merge_target],
                outputs=[merge_output, merge_source, merge_target, rename_old]
            )

            rename_btn.click(
                fn=handle_rename,
                inputs=[rename_old, rename_new],
                outputs=[rename_output, merge_source, merge_target, rename_old]
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

            ollama_model_input = gr.Textbox(
                label="Ollama model",
                value=cfg.get("ollama_model", "llama3.2"),
                placeholder="llama3.2"
            )

            ollama_url_input = gr.Textbox(
                label="Ollama URL (advanced)",
                value=cfg.get("ollama_url", "http://localhost:11434"),
            )

            save_settings_btn = gr.Button("💾 Save Settings")
            settings_output = gr.Markdown()

            def save_settings(name, model, url):
                config.set_value("user_name", name)
                config.set_value("ollama_model", model)
                config.set_value("ollama_url", url)
                return "✅ Settings saved."

            save_settings_btn.click(
                fn=save_settings,
                inputs=[user_name_input, ollama_model_input, ollama_url_input],
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