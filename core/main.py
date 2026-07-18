# Import sahi tarike se (config folder se)
from config.config import is_configured, get_key

class LLMBrain:
    def __init__(self):
        self.active = is_configured()
        self.api_key = get_key()

    async def generate_response(self, prompt: str) -> str:
        if not self.active:
            return "System Offline: API key missing in .env"
        # Yahan tum GLM 5.2 ka API call logic add karoge
        return f"GLM 5.2 Active. Processing: {prompt}"
