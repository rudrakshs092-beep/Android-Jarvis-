You are an expert Python backend developer. Write the complete code for a JARVIS-style voice assistant backend. Produce TWO files:

**File 1: `voice_engine.py`**
- Use `faster-whisper` for Speech-to-Text (continuous microphone listening).
- Use `edge-tts` for Text-to-Speech (convert AI text responses to audio).
- Must be modular: functions for `listen()`, `speak()`, and a callback system.
- Support hot-word detection (e.g., "JARVIS") to trigger listening mode.

**File 2: `app.py` (FastAPI WebSocket Server)**
- Create a FastAPI server with WebSocket endpoint `/ws`.
- WebSocket events to send to frontend (`frontend/index.html`):
  - Send `{"state": "listening"}` when mic is active.
  - Send `{"state": "talking"}` when TTS is playing audio.
  - Send `{"state": "idle"}` when waiting.
  - Send `{"transcript": "..."}` when speech is recognized.
- Also expose a REST endpoint `POST /chat` that accepts `{"message": "..."}` and returns `{"response": "..."}`.
- The server must run on `0.0.0.0:8000` so it works on Android/Termux.

**Requirements:**
- All code must be well-commented (English comments).
- Handle errors gracefully (no crashes on mic failure, no audio).
- Use `asyncio` properly — mic listening and TTS must not block the WebSocket.
- Include a `requirements.txt` section at the bottom.
- Make it production-ready: clean imports, no global state, thread-safe where needed.
- Design it so it can be embedded in an Android WebView app (i.e., runs headless, no GUI).

Write the complete code for both files. Do not skip any section.
