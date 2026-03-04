"""
adapters/claude.py
Claude export adapter — coming in v2.
"""

from adapters.base import BaseAdapter
from pathlib import Path
from typing import List, Dict, Any


class ClaudeAdapter(BaseAdapter):

    def get_name(self) -> str:
        return "Claude"

    def get_export_instructions(self) -> str:
        return "Claude export support coming in v2."

    def load_from_zip(self, zip_path: Path) -> List[Dict[str, Any]]:
        raise NotImplementedError("Claude adapter coming in v2.")