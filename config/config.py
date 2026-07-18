"""
config/config.py
JARVIS Assistant - Universal Configuration Module

Dynamically loads API keys from a .env file without hardcoding variable names.
No external libraries required.
"""

import os
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Path to .env file located next to this config file
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

# Dictionary to hold all loaded environment variables
_config_vars = {}

def _load_env():
    """Read the .env file and load all key-value pairs dynamically."""
    if not os.path.exists(ENV_PATH):
        logger.warning(f".env file not found at {ENV_PATH}")
        return

    try:
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip blank lines and comments
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue

                # Split into key and value
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")

                if key:
                    _config_vars[key] = value
                    # Also export to os.environ so 3rd-party SDKs can use them natively
                    os.environ[key] = value

    except Exception as e:
        logger.error(f"Failed to read .env file: {e}")

# Load variables once when the module is imported
_load_env()


def get_api_key(provider_name: str) -> str | None:
    """
    Dynamically retrieve an API key for a given provider.
    
    Args:
        provider_name (str): The provider name (e.g., 'openai', 'GLM', 'groq').
                             The function will automatically look for 'PROVIDER_KEY'.
                             
    Returns:
        str or None: The API key, or None if not found.
    """
    provider_name = provider_name.strip().upper()
    
    # Generate the expected key name (e.g., "openai" -> "OPENAI_KEY")
    if provider_name.endswith("_KEY"):
        target_key = provider_name
    else:
        target_key = f"{provider_name}_KEY"
        
    # Retrieve the key
    key_value = _config_vars.get(target_key)
    
    if not key_value:
        logger.warning(f"API key for provider '{provider_name}' (looked for '{target_key}') not found.")
        return None
        
    return key_value


def is_active(provider_name: str) -> bool:
    """Check if a specific provider's key is configured."""
    return get_api_key(provider_name) is not None
