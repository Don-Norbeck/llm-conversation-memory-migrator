# OwnYourContext

**Your AI Memory. Your Machine. Your Call.**

[![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-v0.2-orange.svg)](https://github.com/Don-Norbeck/llm-conversation-memory-migrator/releases/tag/v0.2)
[![Local Only](https://img.shields.io/badge/processing-100%25%20local-teal.svg)](#privacy)

---

## The Problem

When you switch AI services you lose months of accumulated context. Every new conversation starts from zero.

The obvious fix — just upload your conversation history — doesn't work. A typical ChatGPT export is 400–500MB of JSON. That's approximately 50 million tokens. Claude's context window is 200K tokens. You can't upload 50 million tokens into a 200K window.

When you switch, you don't just lose your chat history. You lose:

- Months or years of accumulated context about who you are
- Project decisions, code, and artifacts you built together
- Your communication style and preferences the model learned
- Active work threads that are mid-stream

**OwnYourContext fixes that — entirely on your machine.**

---

## The Solution

OwnYourContext compresses your conversation history into a context window that actually fits, then extracts the signals that make a new LLM feel like it already knows you.

1. Export your conversation history from ChatGPT
2. Run the migrator locally — a local AI model summarizes every conversation on your machine
3. 489 conversations become clean markdown files (~200K tokens)
4. Upload to Claude Projects — your context is restored

Tested on 489 real conversations. 98% summarization success rate. 5/5 factual recall on validation tests.

**Your data never leaves your machine.**

---

## What's New in v0.2

- **Biography output** — `get_to_know_me.md` — a warm handoff document your new LLM can use from conversation one
- **Browse tab** — review every conversation before it gets summarized
- **Warm handoff context files** — structured markdown, ready for Claude Projects
- **98% pipeline success** validated across 489 real conversations

[→ View v0.2 Release Notes](https://github.com/Don-Norbeck/llm-conversation-memory-migrator/releases/tag/v0.2)

---

## Privacy First. Always.

This tool was built on a single non-negotiable principle: **your conversations are yours**.

- ✅ Runs 100% on your local machine
- ✅ No internet connection required
- ✅ No accounts, no sign-ups, no API keys required
- ✅ No telemetry, no analytics, no logging to any server
- ✅ No data ever transmitted outside your machine
- ✅ Fully open source — verify every line yourself
- ❌ No cloud processing
- ❌ No shared infrastructure
- ❌ No exceptions

Summarization is powered by **Ollama + mistral-nemo:12b** running locally on your machine. If you prefer to use your own OpenAI or Anthropic API key for higher quality summaries, that option is available — but your data goes only to your own account, never to ours.

---

## Who This Is For

- Someone switching from ChatGPT to Claude (or any LLM to any LLM)
- Anyone who has built up months of context with one AI and doesn't want to start over
- People who care about privacy and want full control of their data
- Power users who want clean, structured project context in their new LLM

**No technical experience required.**

---

## Quickstart

### Step 1 — Install Ollama

Download and install Ollama from [ollama.com](https://ollama.com). It installs like any normal application.

Then open a terminal and pull the local model:

```bash
ollama pull mistral-nemo
```

This downloads the local AI model that powers summarization (~7GB, one time).

### Step 2 — Install OwnYourContext

```bash
git clone https://github.com/Don-Norbeck/llm-conversation-memory-migrator
cd llm-conversation-memory-migrator
pip install -r requirements.txt
python app.py
```

A local webpage will open in your browser. No internet connection is used.

### Step 3 — Export Your Chat History

**From ChatGPT:**

1. Go to chatgpt.com → Settings → Data Controls
2. Click **Export Data**
3. Wait for the email from OpenAI with your download link
4. Download the zip file

### Step 4 — Run the Migration

1. Open the app (`python app.py` if not already running)
2. Click **Load Export** and select your zip file
3. Click **Analyze** and wait (~5–15 minutes depending on history size)
4. Browse conversations — review, select, or deselect before export
5. Click **Export** to generate your context documents

### Step 5 — Import to Claude

1. Go to [claude.ai](https://claude.ai) → Projects → New Project
2. Click **Add Content** → **Upload Files**
3. Upload `get_to_know_me.md` first — this is your warm handoff document
4. Add any topic context files for the domains you're working in
5. Start chatting — Claude now has your full context

---

## Output Files

**`get_to_know_me.md`** — Your warm handoff document. Upload this first to any new LLM. Contains:
- Who you are — name, role, background, credentials
- What you work on — active projects, domains, key terms (signal-tier only)
- How you work — communication style, format preferences, working patterns
- Don't do this — explicit negative preferences from your correction history
- Your words — 2–5 verbatim phrases that capture your voice and framing

**`[topic]_context.md`** — One file per topic domain. Deeper detail for domain-specific work. Upload the relevant files when starting work in that area.

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10, macOS 12, Ubuntu 20.04 | Windows 11, macOS 14 |
| RAM | 8GB | 16GB+ |
| Storage | 10GB free | 20GB free |
| GPU | Not required | NVIDIA/AMD GPU speeds up processing |
| Internet | Not required | Not required |

---

## Supported Services

| Source | Status |
|--------|--------|
| ChatGPT | ✅ Supported |
| Claude | 🔜 v0.3 |
| Gemini | 🔜 v0.3 |
| Microsoft Copilot | 🔜 Planned |

| Target | Status |
|--------|--------|
| Claude Projects | ✅ Supported |
| ChatGPT Memory | 🔜 v0.3 |
| Gemini | 🔜 Planned |

---

## Roadmap

### v0.1 — ChatGPT → Claude ✅ Shipped
- [x] ChatGPT export parser
- [x] Ollama-powered local summarization
- [x] Auto topic classification into 10 buckets
- [x] Interactive review UI (Gradio)
- [x] Bucket merge and rename
- [x] Claude-ready markdown export
- [x] Temp file cleanup — no conversation data left on disk
- [x] Tested on 489 real conversations — 98% success rate, 5/5 recall

### v0.2 — Quality & Biography ✅ Shipped — March 2026
- [x] Browse tab — view, sort, and filter all conversations after upload
- [x] Biography output — `get_to_know_me.md` warm handoff document
- [x] Warm handoff context files — structured per-topic markdown
- [x] WHO/WHAT/HOW signal extraction per conversation
- [x] Name recognition — user name injected correctly throughout biography
- [x] Cyberpunk UI — OwnYourContext brand applied to Gradio app
- [x] 98% pipeline success validated on full 489-conversation dataset

### v0.3 — UI alignment, Deeper User Context  🔄 In Progress
- [ ] Four-axis extraction redesign — WHO/WHAT/HOW/WHY with tier classification
- [ ] Cross-conversation synthesis pass — pattern detection across full history
- [ ] User editing layer — per-conversation promote/demote/delete controls
- [ ] Profile section editing — edit, extend, or remove any auto-extracted field
- [ ] Sensitive content flagging — health, financial, family details flagged for review
- [ ] Selective analysis — wire Browse tab selection to analyzer
- [ ] WHY section — user-authored motivation layer with guided prompts
- [ ] Direct Claude Projects API upload
- [ ] Delta updates — re-run and merge new conversations only
- [ ] Test mode UX fix — reliable checkbox, clear conversation cap control

### v1.0 — Packaged for Everyone
- [ ] Windows installer (.exe)
- [ ] Mac installer (.app)
- [ ] One-click Ollama and model setup
- [ ] No terminal required
- [ ] Onboarding interview — structured questions generate your context profile automatically

---

## Contributing

Contributions welcome, especially:

- **New source adapters** — add support for exporting from other LLM services
- **New target adapters** — add support for importing to other LLM services
- **Language translations** — help make this accessible globally
- **UX improvements** — make it even simpler for non-technical users

See [docs/adding-adapters.md](docs/adding-adapters.md) for how to add a new source or target adapter.

---

## License

MIT — free to use, modify, and distribute.

---

## About

Built by [Don Norbeck](https://darkaidefense.com) — because your accumulated context belongs to you, not to the platform you happened to use when you built it.

[ownyourcontext.com](https://ownyourcontext.com) · [darkaidefense.com](https://darkaidefense.com) · [GitHub](https://github.com/Don-Norbeck/llm-conversation-memory-migrator)

---

*If this tool helped you, star the repo and share it with someone else making the switch.*
