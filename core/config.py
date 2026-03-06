"""
core/config.py
Local configuration management.
All settings stored on user's machine only.
No external API calls. Ever.
"""

import json
from pathlib import Path
from typing import Any, Optional

# ── Config file location ─────────────────────────────────────────────────────

CONFIG_DIR = Path.home() / ".llm-migrator"
CONFIG_FILE = CONFIG_DIR / "config.json"

# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULTS = {
    "ollama_model": "llama3.2",
    "ollama_url": "http://localhost:11434",
    "output_dir": str(Path.home() / "Documents" / "LLMMigrator" / "output"),
    "user_name": "",
    "last_export_path": "",
    "summary_style": "concise",
    "test_mode": False,
    "test_mode_n": 10,
}


# ── Load / Save ───────────────────────────────────────────────────────────────

def load_config() -> dict:
    """
    Load config from disk. Returns defaults if no config file exists.
    Never fails — always returns a valid config dict.
    """
    config = DEFAULTS.copy()
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            config.update(saved)
        except Exception as e:
            print(f"Warning: Could not load config: {e}. Using defaults.")
    return config


def save_config(config: dict) -> bool:
    """
    Save config to disk.
    Returns True on success, False on failure.
    """
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Warning: Could not save config: {e}")
        return False


def get(key: str, default: Any = None) -> Any:
    """Get a single config value."""
    config = load_config()
    return config.get(key, default)


def set_value(key: str, value: Any) -> bool:
    """Set a single config value and save."""
    config = load_config()
    config[key] = value
    return save_config(config)


def reset_to_defaults() -> bool:
    """Reset all config to defaults."""
    return save_config(DEFAULTS.copy())


# ── Ollama helpers ────────────────────────────────────────────────────────────

def get_ollama_config() -> dict:
    """Return Ollama connection config."""
    config = load_config()
    return {
        "model": config["ollama_model"],
        "url": config["ollama_url"],
    }


def get_output_dir() -> Path:
    """Get the configured output directory as a Path object."""
    return Path(get("output_dir", DEFAULTS["output_dir"]))


def set_output_dir(path: str) -> bool:
    """Set the output directory."""
    return set_value("output_dir", str(path))


def get_user_name() -> Optional[str]:
    """Get the configured user name."""
    name = get("user_name", "")
    return name if name.strip() else None
