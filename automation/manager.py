"""
automation/manager.py
JARVIS Assistant - Automation Manager

Routes high-level intents to specific hardware control functions.
Designed to be easily imported and called by core/brain.py.
"""

import logging
from typing import Any, Dict, Optional
from automation.android_controls import (
    toggle_flashlight,
    set_volume,
    get_battery_info,
    get_device_info,
    run_shell_command
)

# Configure logger
logger = logging.getLogger(__name__)

# Map intent strings to their respective async functions
ACTION_MAP = {
    "toggle_flashlight": toggle_flashlight,
    "set_volume": set_volume,
    "get_battery_info": get_battery_info,
    "get_device_info": get_device_info,
    "run_shell_command": run_shell_command,
}


async def execute_action(intent: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """
    Executes a mapped automation action based on the provided intent.

    Args:
        intent (str): The action to perform (e.g., 'toggle_flashlight').
        params (dict, optional): Dictionary of arguments to pass to the function. 
                                  Defaults to None (empty dict).

    Returns:
        Any: The result of the executed function (usually a string or dict), 
             or an error/unknown command string.
    """
    if params is None:
        params = {}

    # 1. Check if the intent exists in our map
    if intent not in ACTION_MAP:
        logger.warning(f"Unknown intent requested: {intent}")
        return "Unknown command"

    func = ACTION_MAP[intent]

    # 2. Attempt to execute the function with error handling
    try:
        # Unpack the params dictionary as keyword arguments
        result = await func(**params)
        return result
    except TypeError as e:
        # Catches errors if the wrong params are passed to a function
        logger.error(f"Parameter mismatch for intent '{intent}': {e}")
        return f"Error: Invalid parameters for {intent}."
    except Exception as e:
        # Catch-all for any unexpected errors during execution
        logger.error(f"Execution failed for intent '{intent}': {e}")
        return f"Error: Failed to execute {intent}."
