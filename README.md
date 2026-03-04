# llm-conversation-memory-migrator

When you switch AI services you lose months of accumulated context. This fixes that. Local-only, open source, works for anyone — starting with ChatGPT → Claude.  No cloud. No data sharing. Your context/conversations belong to you.

# llm-conversation-memory-migrator



> Move your mind, not just your messages.



A fully local, privacy-first tool that helps anyone migrate their conversation history and accumulated context from one LLM service to another — starting with ChatGPT → Claude.



\*\*Your data never leaves your machine. Ever.\*\*



---



\## Why This Exists



When you switch LLM services, you don't just lose your chat history. You lose:



\- Months or years of accumulated context about who you are

\- Project decisions, code, and artifacts you built together

\- Your communication style and preferences the model learned

\- Active work threads that are mid-stream



This tool fixes that. It reads your export, understands what matters, and produces clean context documents you can drop straight into your new LLM — so you pick up where you left off.



---



\## Privacy First. Always.



This tool was built on a single non-negotiable principle: \*\*your conversations are yours\*\*.



\- ✅ Runs 100% on your local machine

\- ✅ No internet connection required

\- ✅ No accounts, no sign-ups, no API keys required (optional for power users)

\- ✅ No telemetry, no analytics, no logging to any server

\- ✅ No data ever transmitted outside your machine

\- ✅ Fully open source — verify every line yourself

\- ❌ No cloud processing

\- ❌ No shared infrastructure

\- ❌ No exceptions



Summarization is powered by \*\*Ollama + Llama 3.2\*\* running locally on your machine. If you prefer to use your own OpenAI or Anthropic API key for higher quality summaries, that option is available — but your data goes only to your own account, never to ours.



---



\## Who This Is For



\- Someone switching from ChatGPT to Claude (or any LLM to any LLM)

\- Anyone who has built up months of context with one AI and doesn't want to start over

\- People who care about privacy and want full control of their data

\- Power users who want clean, structured project context in their new LLM

\- \*\*No technical experience required\*\*



---



\## How It Works



```

1\. Export your chat history from your current LLM service

2\. Open llm-conversation-memory-migrator

3\. Drop in your export file

4\. Click Analyze — local AI reads and summarizes your conversations

5\. Review auto-detected topics, rename or merge as needed

6\. Click Export — get clean context documents per topic

7\. Upload to your new LLM service and pick up where you left off

```



Total time: \*\*15–30 minutes\*\* for most users.



---



\## Supported Services



| Source | Status |

|--------|--------|

| ChatGPT | ✅ Supported (v1) |

| Claude | 🔜 Coming in v2 |

| Gemini | 🔜 Coming in v2 |

| Microsoft Copilot | 🔜 Planned |



| Target | Status |

|--------|--------|

| Claude Projects | ✅ Supported (v1) |

| ChatGPT Memory | 🔜 Coming in v2 |

| Gemini | 🔜 Planned |



---



\## Quickstart



\### Step 1 — Install Ollama



Download and install Ollama from \[ollama.com](https://ollama.com). It installs like any normal application.



Then open a terminal and run:



```bash

ollama pull llama3.2

```



This downloads the local AI model that powers summarization (~2GB, one time).



\### Step 2 — Install the Migrator



```bash

git clone https://github.com/\[your-handle]/llm-conversation-memory-migrator

cd llm-conversation-memory-migrator

pip install -r requirements.txt

python app.py

```



A local webpage will open in your browser. That's the app — no internet connection is used.



\### Step 3 — Export Your Chat History



\*\*From ChatGPT:\*\*

1\. Go to chatgpt.com → Settings → Data Controls

2\. Click \*\*Export Data\*\*

3\. Wait for the email from OpenAI with your download link

4\. Download the zip file



\### Step 4 — Run the Migration



1\. Open the app (if not already open: `python app.py`)

2\. Click \*\*Upload Export\*\* and select your zip file

3\. Click \*\*Analyze\*\* and wait (~5–15 minutes depending on history size)

4\. Review the auto-detected topic buckets

5\. Rename, merge, or split topics as needed

6\. Click \*\*Export\*\* to generate your context documents



\### Step 5 — Import to Claude



1\. Go to \[claude.ai](https://claude.ai) → Projects → New Project

2\. Name it after one of your topic buckets

3\. Click \*\*Add Content\*\* → \*\*Upload Files\*\*

4\. Select the exported markdown files for that topic

5\. Repeat for each topic

6\. Start chatting — Claude now has your full context



---



\## Output Format



For each topic bucket the tool generates:



\*\*`\[topic]\_context.md`\*\* — A structured summary containing:

\- Topic overview and key themes

\- Important decisions and conclusions reached

\- Artifacts created (code, documents, frameworks)

\- Open threads and unresolved questions

\- Your preferences and communication style for this domain

\- Chronological timeline of key milestones



\*\*`master\_context.md`\*\* — A single document summarizing who you are, your projects, preferences, and accumulated context across all topics. Useful as a universal system prompt or project briefing.



---



\## System Requirements



| Component | Minimum | Recommended |

|-----------|---------|-------------|

| OS | Windows 10, macOS 12, Ubuntu 20.04 | Windows 11, macOS 14 |

| RAM | 8GB | 16GB+ |

| Storage | 5GB free | 10GB free |

| GPU | Not required | Any NVIDIA/AMD GPU speeds up processing |

| Internet | Not required | Not required |



---



\## Power User Options



If you want higher-quality summaries and have your own API key:



```

Settings → Summarization Backend → Choose:

&nbsp; ● Local (Ollama) — default, fully private

&nbsp; ○ OpenAI (your key) — faster, higher quality

&nbsp; ○ Anthropic (your key) — fastest, highest quality

```



Your API key is stored locally in a config file on your machine. It is never transmitted anywhere except directly to your own API account.



---



\## Contributing



This project welcomes contributions, especially:



\- \*\*New source adapters\*\* — add support for exporting from other LLM services

\- \*\*New target adapters\*\* — add support for importing to other LLM services

\- \*\*Language translations\*\* — help make this accessible globally

\- \*\*UX improvements\*\* — make it even simpler for non-technical users



See \[docs/adding-adapters.md](docs/adding-adapters.md) for how to add a new LLM source or target.



---



\## Roadmap



\*\*v1.0 — ChatGPT → Claude\*\*

\- \[x] ChatGPT export parser

\- \[ ] Ollama-powered local summarization

\- \[ ] Auto topic classification

\- \[ ] Interactive review UI (Gradio)

\- \[ ] Claude-ready markdown export

\- \[ ] Master context document generation

\- \[ ] Windows + Mac packaging (.exe / .app)



\*\*v2.0 — Any LLM → Any LLM\*\*

\- \[ ] Claude export parser

\- \[ ] Gemini export parser

\- \[ ] Direct Claude Projects API upload

\- \[ ] ChatGPT memory import format

\- \[ ] Multi-language support



---



\## License



MIT — free to use, modify, and distribute.



---



\## About



Built by \[Don Norbeck](https://darkaidefense.com) — because your accumulated context belongs to you, not to the platform you happened to use when you built it.



---



\*If this tool helped you, consider starring the repo and sharing it with someone else making the switch.\*

