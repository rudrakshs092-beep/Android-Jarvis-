import os
from pathlib import Path

# `.env` file ko root directory mein dhundhne ke liye
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

def _load_key():
    if not ENV_PATH.exists(): return None
    with open(ENV_PATH, "r") as f:
        for line in f:
            if "=" in line:
                key, val = line.strip().split("=", 1)
                if key.strip() in ["API_KEY", "GLM_KEY", "KEY"]:
                    return val.strip().strip('"')
    return None

API_KEY = _load_key()
is_active = API_KEY is not None

def is_configured(): return is_active
def get_key(): return API_KEY
