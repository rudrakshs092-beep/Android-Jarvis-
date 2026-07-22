You are an expert JavaScript developer. Write a lightweight, robust, production-ready JavaScript snippet that connects my existing `frontend/index.html` canvas UI to a FastAPI WebSocket backend running at `ws://127.0.0.1:8000/ws`.

**Context:**
- My `index.html` already has a canvas-based UI with a `StateMachine` object.
- `StateMachine.setState(state)` accepts: "listening", "talking", "idle"
- The backend sends JSON events over WebSocket.

**File to produce: `frontend/ws_connector.js`**

**Exact Requirements:**

1. **WebSocket Connection:**
   - Connect to `ws://127.0.0.1:8000/ws`
   - Use `new WebSocket()` API (no libraries).

2. **Auto-Reconnect:**
   - If connection drops (close/error), automatically reconnect every 3 seconds using exponential backoff.
   - Log reconnection attempts to `console.log()`.
   - Cap max retries or use infinite retries with a 3-second interval.

3. **State Updates:**
   - On receiving `{"state": "listening"}` → call `StateMachine.setState("listening")`
   - On receiving `{"state": "talking"}` → call `StateMachine.setState("talking")`
   - On receiving `{"state": "idle"}` → call `StateMachine.setState("idle")`

4. **Transcript Display:**
   - On receiving `{"transcript": "..."}` → display it in a UI HUD element (assume there's a `document.getElementById("hud-transcript")`) and also `console.log()` it.

5. **Two-Way Communication:**
   - Provide a function `sendMessage(message)` that sends `JSON.stringify({message: message})` to the backend.

6. **Quality Requirements:**
   - Zero memory leaks — clean up event listeners on disconnect.
   - Use `async/await` or `.then()` cleanly — no callback hell.
   - Well-commented (English comments explaining each block).
   - Wrap in an IIFE or module pattern so it doesn't pollute global scope.
   - No external dependencies (pure vanilla JS).
   - Must work in mobile WebView / Android Chrome.

7. **HTML Integration:**
   - Show how to include this script in `index.html` (just one `<script>` tag line).

**Constraints:**
- Do NOT rewrite `StateMachine` — assume it already exists.
- Do NOT use any npm packages or build tools.
- Code must be complete — no placeholders like "// TODO" or "// your logic here".
- Minimum ~60 lines, production-grade quality.

Write the complete `ws_connector.js` file. Show the exact `<script>` tag to add in `index.html`.
