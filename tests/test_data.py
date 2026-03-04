"""
tests/test_data.py
Small synthetic ChatGPT export for fast local testing.
No real user data — safe to commit to GitHub.
"""

import json
from pathlib import Path

TEST_CONVERSATIONS = [
    {
        "id": "test-001",
        "title": "Dark AI Defense Logo",
        "create_time": 1720000000,
        "update_time": 1720001000,
        "mapping": {
            "msg-1": {
                "message": {
                    "author": {"role": "user"},
                    "create_time": 1720000000,
                    "content": {"parts": ["I need a logo for Dark AI Defense. Minimalist, cybersecurity inspired, dark theme."]}
                }
            },
            "msg-2": {
                "message": {
                    "author": {"role": "assistant"},
                    "create_time": 1720000100,
                    "content": {"parts": ["Here are some logo concepts for Dark AI Defense. A shield with a circuit pattern would convey cybersecurity. Dark navy and electric blue palette. Minimalist sans-serif typography."]}
                }
            }
        }
    },
    {
        "id": "test-002",
        "title": "Python Script for Data Processing",
        "create_time": 1720100000,
        "update_time": 1720101000,
        "mapping": {
            "msg-1": {
                "message": {
                    "author": {"role": "user"},
                    "create_time": 1720100000,
                    "content": {"parts": ["Write a Python script to parse CSV files and output JSON."]}
                }
            },
            "msg-2": {
                "message": {
                    "author": {"role": "assistant"},
                    "create_time": 1720100100,
                    "content": {"parts": ["Here's a Python script using pandas to parse CSV and convert to JSON. import pandas as pd. df = pd.read_csv('input.csv'). df.to_json('output.json')."]}
                }
            }
        }
    },
    {
        "id": "test-003",
        "title": "Resume Update for AI Leadership Role",
        "create_time": 1720200000,
        "update_time": 1720201000,
        "mapping": {
            "msg-1": {
                "message": {
                    "author": {"role": "user"},
                    "create_time": 1720200000,
                    "content": {"parts": ["Help me update my resume for a Head of AI position. I have 7 patents in cloud architecture."]}
                }
            },
            "msg-2": {
                "message": {
                    "author": {"role": "assistant"},
                    "create_time": 1720200100,
                    "content": {"parts": ["For a Head of AI role, lead with your patent portfolio and scale achievements. Highlight the 0 to 1B platform scaling experience. Add an AI Strategy section."]}
                }
            }
        }
    },
    {
        "id": "test-004",
        "title": "Stowe Ski Trip Planning",
        "create_time": 1720300000,
        "update_time": 1720301000,
        "mapping": {
            "msg-1": {
                "message": {
                    "author": {"role": "user"},
                    "create_time": 1720300000,
                    "content": {"parts": ["Planning a ski trip to Stowe Vermont. Best runs for advanced skiers?"]}
                }
            },
            "msg-2": {
                "message": {
                    "author": {"role": "assistant"},
                    "create_time": 1720300100,
                    "content": {"parts": ["For advanced skiers at Stowe, try Goat and Liftline on the Front Four. National and Starr are also excellent. Go early before the crowds hit the groomers."]}
                }
            }
        }
    },
    {
        "id": "test-005",
        "title": "Ollama Local AI Setup",
        "create_time": 1720400000,
        "update_time": 1720401000,
        "mapping": {
            "msg-1": {
                "message": {
                    "author": {"role": "user"},
                    "create_time": 1720400000,
                    "content": {"parts": ["How do I set up Ollama with Llama 3 on my local machine?"]}
                }
            },
            "msg-2": {
                "message": {
                    "author": {"role": "assistant"},
                    "create_time": 1720400100,
                    "content": {"parts": ["Install Ollama from ollama.com. Then run ollama pull llama3.2 in your terminal. It will download the model locally. Run ollama run llama3.2 to test it."]}
                }
            }
        }
    }
]


def create_test_zip(output_path: Path) -> Path:
    """
    Create a minimal test zip file mimicking a ChatGPT export.
    Safe to use for testing — no real user data.
    """
    import zipfile
    import tempfile
    import os

    zip_path = output_path / "test_export.zip"
    
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "conversations-000.json",
            json.dumps(TEST_CONVERSATIONS, indent=2)
        )
        zf.writestr("user.json", json.dumps({"name": "Test User"}))

    print(f"Test export created: {zip_path}")
    return zip_path


if __name__ == "__main__":
    output = Path("tests")
    output.mkdir(exist_ok=True)
    create_test_zip(output)
    print("Done. Upload tests/test_export.zip to the app to test.")