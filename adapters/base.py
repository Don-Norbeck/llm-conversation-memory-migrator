"""
adapters/base.py
Abstract base class for LLM export adapters.
Extend this to add support for new LLM services.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any


class BaseAdapter(ABC):
    """
    Base class for all LLM export adapters.
    
    To add a new source LLM:
    1. Create a new file in adapters/ (e.g. adapters/gemini.py)
    2. Subclass BaseAdapter
    3. Implement load_from_zip()
    4. Register in app.py
    """

    @abstractmethod
    def load_from_zip(self, zip_path: Path) -> List[Dict[str, Any]]:
        """
        Parse an export zip file into normalized conversation dicts.
        
        Returns:
            List of dicts with keys:
            - id: str
            - title: str
            - created: str
            - updated: str
            - messages: List[{role, text, timestamp}]
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the display name of this adapter e.g. 'ChatGPT'"""
        pass

    @abstractmethod
    def get_export_instructions(self) -> str:
        """Return user-facing instructions for exporting from this service."""
        pass