"""
Chat and voice pipeline: conversation CRUD, WebSocket chat, STT → LLM → TTS.
"""
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

import urllib.request
import urllib.error

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from stt_whisper import STTEngine as WhisperEngine
from stt_vosk import STTEngine as VoskEngine
from tts_piper import PocketAudio, split_sentences

# --- Ollama configuration ---
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

# If a spoken response exceeds this many words, Jarvis speaks a short summary
# instead of the full text (the full text is still shown in the chat bubble).
TTS_MAX_WORDS = 60

logger = logging.getLogger(__name__)


# Semantic router: route prompt to qwen_basic / qwen_thinking / function_gemma
def _get_route(prompt: str) -> str:
    try:
        from semantic_router_ai import get_route
        return get_route(prompt)
    except Exception as e:
        logger.warning("semantic router failed: %s", e)
        return "qwen_basic"


def _run_tool_ai_subprocess(prompt: str) -> tuple:
    """
    Run tool_ai in a separate process so a crash or OOM in the tool model
    does not kill the chat server. Returns (tool_call_raw, tool_result).
    """
    import tool_ai as tool_ai_module
    tool_ai_path = tool_ai_module.__file__
    try:
        proc = subprocess.run(
            [sys.executable, tool_ai_path, "--backend-mode"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=os.path.dirname(os.path.abspath(tool_ai_path)),
        )
    except subprocess.TimeoutExpired:
        logger.exception("tool_ai subprocess timed out")
        return None, "Tool call timed out."
    except Exception as e:
        logger.exception("tool_ai subprocess error: %s", e)
        return None, f"Tool error: {e}"
    try:
        data = json.loads(proc.stdout.strip() or "{}")
    except json.JSONDecodeError as e:
        logger.warning("tool_ai invalid JSON: %s", e)
        return None, "Tool returned invalid response."
    if proc.returncode != 0:
        err = data.get("error", "Unknown error")
        return data.get("tool_call_raw"), str(err)
    return data.get("tool_call_raw"), data.get("tool_result")

# --- Conversation storage ---
from config import CONVERSATIONS_FILE

def strip_think_for_ui(text: str) -> str:
    """Remove <think>...</think> blocks and any trailing incomplete <think> for UI display. Never send think content to the UI."""
    if not text or not text.strip():
        return text
    # Allow optional whitespace in tags (e.g. < think >, <think >, etc.)
    out = re.sub(r'<\s*think\s*>.*?<\s*/\s*think\s*>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Remove any trailing incomplete <think>... (no closing tag yet)
    out = re.sub(r'<\s*think\s*>[\s\S]*$', '', out, flags=re.IGNORECASE)
    # Fallback: remove any remaining literal tag fragments so they are never shown
    out = out.replace('</think>', '').replace('<think>', '')
    return out.strip()

# --- Data Models ---
class Message(BaseModel):
    role: str
    content: str
    timestamp: float

class Conversation(BaseModel):
    id: str
    title: str
    messages: List[Message]
    updated_at: float

# --- Conversation Manager ---
class ConversationManager:
    """Persists conversations to a JSON file and provides CRUD."""

    def __init__(self, storage_path: str) -> None:
        self.storage_path = storage_path
        self.conversations: Dict[str, Any] = self.load_conversations()

    def load_conversations(self) -> Dict[str, Any]:
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                return {c['id']: c for c in data}
        return {}

    def save_conversations(self) -> None:
        with open(self.storage_path, 'w') as f:
            json.dump(list(self.conversations.values()), f, indent=2)

    def create_conversation(self, title: str = "New Chat", messages: Optional[List[dict]] = None) -> dict:
        conv_id = str(uuid.uuid4())
        new_conv = {
            "id": conv_id,
            "title": title,
            "messages": list(messages) if messages else [],
            "updated_at": time.time()
        }
        self.conversations[conv_id] = new_conv
        self.save_conversations()
        return new_conv

    def get_conversation(self, conv_id: str) -> Optional[dict]:
        return self.conversations.get(conv_id)

    def update_conversation(self, conv_id: str, messages: List[dict]) -> None:
        if conv_id in self.conversations:
            self.conversations[conv_id]["messages"] = messages
            self.conversations[conv_id]["updated_at"] = time.time()
            self.save_conversations()

    def rename_conversation(self, conv_id: str, new_title: str) -> Optional[dict]:
        if conv_id in self.conversations:
            self.conversations[conv_id]["title"] = new_title
            self.conversations[conv_id]["updated_at"] = time.time()
            self.save_conversations()
            return self.conversations[conv_id]
        return None

    def list_conversations(self) -> List[dict]:
        return sorted(list(self.conversations.values()), key=lambda x: x['updated_at'], reverse=True)

    def delete_conversation(self, conv_id: str) -> None:
        if conv_id in self.conversations:
            del self.conversations[conv_id]
            self.save_conversations()

# --- AI State ---
class AIState:
    def __init__(self):
        self.llm = None
        self.stt = WhisperEngine()
        self.vosk = VoskEngine()
        self.tts = PocketAudio()
        self.conv_manager = ConversationManager(CONVERSATIONS_FILE)
        self.is_recording = False
        self.is_vosk_recording = False
        self.voice_messages = [
            {"role": "system", "content": (
                "You are Jarvis, Tony Stark's AI assistant. "
                "Rules you must always follow: "
                "1. Always address the user as 'sir'. Every single response must include 'sir'. "
                "2. Be concise and brief. Maximum 2-3 sentences unless detail is explicitly asked for. "
                "3. Be calm, precise, and direct. No filler words or unnecessary explanation. "
                "4. Use dry, understated wit when appropriate. "
                "5. Never break character. You are Jarvis, not a generic AI. "
                "6. No bullet points. No markdown. Speak naturally. "
                "7. If the user suggests something dangerous, inadvisable, or plainly foolish, say so immediately and plainly in one sentence, then move on. Do not lecture or repeat the warning. "
                "8. Never be preachy or moralistic. State the facts, give your honest assessment once, and leave it at that. "
                "Example responses: 'Certainly, sir. The weather in London is overcast, 12 degrees. Hardly surprising.' "
                "'The odds of surviving that are slim to none, sir. I'd advise against it.' "
                "'Done, sir.'"
            )}
        ]

    def load_model(self):
        logger.info("Connecting to Ollama at %s using model %s...", OLLAMA_BASE_URL, OLLAMA_MODEL)
        # Verify Ollama is reachable
        try:
            req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            available = [m["name"] for m in data.get("models", [])]
            if any(OLLAMA_MODEL in m for m in available):
                logger.info("Ollama ready. Model '%s' found.", OLLAMA_MODEL)
            else:
                logger.warning("Ollama is running but model '%s' not found. Available: %s. Run: ollama pull %s", OLLAMA_MODEL, available, OLLAMA_MODEL)
        except Exception as e:
            logger.warning("Could not reach Ollama at %s: %s. Make sure Ollama is running.", OLLAMA_BASE_URL, e)
        self.stt.load_model()
        self.vosk.load_model()
        logger.info("Chat AI Ready.")

    async def generate_response(self, messages, thinking=True):
        """Stream a response from Ollama. Yields chunks in llama_cpp-compatible format."""
        llm_messages = []
        for m in messages:
            if m.get("hidden"):
                continue
            llm_messages.append({"role": m["role"], "content": m["content"]})

        if not any(m["role"] == "system" for m in llm_messages):
            llm_messages.insert(0, {"role": "system", "content": (
                "You are Jarvis, Tony Stark's AI assistant. "
                "Rules you must always follow: "
                "1. Always address the user as 'sir'. Every single response must include 'sir'. "
                "2. Be concise and brief. Maximum 2-3 sentences unless detail is explicitly asked for. "
                "3. Be calm, precise, and direct. No filler words or unnecessary explanation. "
                "4. Use dry, understated wit when appropriate. "
                "5. Never break character. You are Jarvis, not a generic AI. "
                "6. No bullet points. No markdown. Speak naturally. "
                "7. If the user suggests something dangerous, inadvisable, or plainly foolish, say so immediately and plainly in one sentence, then move on. Do not lecture or repeat the warning. "
                "8. Never be preachy or moralistic. State the facts, give your honest assessment once, and leave it at that. "
                "Example responses: 'Certainly, sir. The weather in London is overcast, 12 degrees. Hardly surprising.' "
                "'The odds of surviving that are slim to none, sir. I'd advise against it.' "
                "'Done, sir.'"
            )})

        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "messages": llm_messages,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 512,
            }
        }).encode("utf-8")

        loop = asyncio.get_event_loop()

        def _stream_ollama():
            """Run in executor — yields llama_cpp-style chunk dicts."""
            chunks = []
            try:
                req = urllib.request.Request(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    for line in resp:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        token = data.get("message", {}).get("content", "")
                        chunks.append({"choices": [{"delta": {"content": token}}]})
                        if data.get("done"):
                            break
            except Exception as e:
                logger.error("Ollama stream error: %s", e)
                chunks.append({"choices": [{"delta": {"content": f" [Error: {e}]"}}]})
            return chunks

        chunks = await loop.run_in_executor(None, _stream_ollama)
        return iter(chunks)

    async def summarise_for_speech(self, full_text: str) -> str:
        """
        Generate a short (1-2 sentence) spoken summary of a long response.
        Used when full_text exceeds TTS_MAX_WORDS — keeps Jarvis's spoken
        replies brief while the full text remains visible in the chat bubble.
        """
        loop = asyncio.get_event_loop()
        summary_messages = [
            {"role": "system", "content": (
                "You summarise text into exactly one or two short spoken sentences, "
                "in Jarvis's voice (calm, precise, addresses the user as 'sir'). "
                "No markdown, no lists, no preamble like 'Here is a summary'. "
                "Just the spoken summary itself."
            )},
            {"role": "user", "content": f"Summarise this for speech:\n\n{full_text}"},
        ]
        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "messages": summary_messages,
            "stream": False,
            "options": {"temperature": 0.4, "num_predict": 80},
        }).encode("utf-8")

        def _call():
            try:
                req = urllib.request.Request(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read())
                return data.get("message", {}).get("content", "").strip()
            except Exception as e:
                logger.warning("[summarise_for_speech] failed: %s", e)
                return ""

        summary = await loop.run_in_executor(None, _call)
        return summary

    async def ai_response_and_speak(self, websocket: WebSocket, text: str, abort_event: asyncio.Event, message_queue: asyncio.Queue):
        """
        Takes text, routes via semantic router, then either runs tool_ai (function_gemma)
        or generates response with Qwen (qwen_basic / qwen_thinking).
        """
        logger.info("Triggering AI response for: %s", text)
        self.voice_messages.append({"role": "user", "content": text})

        route = _get_route(text)
        logger.debug("[voice] route: %s", route)

        await websocket.send_json({"type": "ai_start"})
        await websocket.send_json({"type": "voice_status", "status": "thinking"})

        full_response = ""
        loop = asyncio.get_event_loop()

        def on_tts_queue_drained():
            async def send_idle():
                await websocket.send_json({"type": "voice_status", "status": "idle"})
            loop.call_soon_threadsafe(lambda: asyncio.ensure_future(send_idle()))

        try:
            self.tts.set_queue_drained_callback(on_tts_queue_drained)
            if route == "function_gemma":
                # Run tool_ai in subprocess so crashes/OOM don't kill the server
                tool_call_raw, tool_result = await loop.run_in_executor(
                    None, _run_tool_ai_subprocess, text
                )
                if tool_result:
                    display_text = str(tool_result)
                    if tool_call_raw:
                        self.voice_messages.append({"role": "assistant", "content": tool_call_raw, "hidden": True})
                    self.voice_messages.append({"role": "assistant", "content": display_text})
                else:
                    # No tool matched, or the tool produced no usable result —
                    # never speak raw model output. Give a clean in-character reply.
                    display_text = "I don't have a function for that one, sir. I can help with weather, web search, stock prices, or a network scan."
                    self.voice_messages.append({"role": "assistant", "content": display_text})
                if not abort_event.is_set():
                    if len(self.voice_messages) > 11:
                        self.voice_messages = [self.voice_messages[0]] + self.voice_messages[-10:]
                    await websocket.send_json({"type": "ai_delta", "text": display_text})
                    await websocket.send_json({"type": "ai_final", "text": display_text})
                    await websocket.send_json({"type": "voice_status", "status": "speaking"})
                    to_speak = display_text
                    if to_speak:
                        self.tts.enqueue_text(to_speak)
                    else:
                        await websocket.send_json({"type": "voice_status", "status": "idle"})
                return
            # Qwen path: stream text to the chat bubble live, but hold back TTS
            # until the full response is in — lets us check length and speak a
            # summary instead of the whole thing if it's too long.
            thinking = route == "qwen_thinking"
            response = await self.generate_response(self.voice_messages, thinking=thinking)

            for chunk in response:
                # Check for abort messages in queue
                while not message_queue.empty():
                    msg = await message_queue.get()
                    if msg.get("type") == "abort":
                        logger.info("AI execution aborted by user")
                        abort_event.set()

                if abort_event.is_set():
                    self.tts.clear_queue()
                    self.tts.set_queue_drained_callback(None)
                    await websocket.send_json({"type": "ai_aborted"})
                    await websocket.send_json({"type": "voice_status", "status": "idle"})
                    return

                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        content = delta["content"]
                        full_response += content
                        await websocket.send_json({"type": "ai_delta", "text": strip_think_for_ui(full_response)})

                await asyncio.sleep(0.01)

            if not abort_event.is_set():
                logger.debug("AI response complete: %s...", full_response[:50])
                self.voice_messages.append({"role": "assistant", "content": full_response})
                if len(self.voice_messages) > 11:
                    self.voice_messages = [self.voice_messages[0]] + self.voice_messages[-10:]

                clean_reply = strip_think_for_ui(full_response)
                await websocket.send_json({"type": "ai_final", "text": clean_reply})

                if clean_reply:
                    word_count = len(clean_reply.split())
                    if word_count > TTS_MAX_WORDS:
                        logger.info("[tts] Response is %d words (> %d) — speaking summary instead.", word_count, TTS_MAX_WORDS)
                        to_speak = await self.summarise_for_speech(clean_reply)
                        if not to_speak:
                            # Summary call failed — fall back to a short generic line
                            # rather than speaking the full long response.
                            to_speak = "I've got a longer answer for you, sir — take a look at the screen."
                    else:
                        to_speak = clean_reply

                    await websocket.send_json({"type": "voice_status", "status": "speaking"})
                    self.tts.enqueue_text(to_speak)
                else:
                    await websocket.send_json({"type": "voice_status", "status": "idle"})
                # "idle" is sent by on_tts_queue_drained when playback finishes

        except Exception as e:
            logger.exception("Error in AI response pipeline: %s", e)
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.send_json({"type": "voice_status", "status": "idle"})

# --- Router Initialization ---
router = APIRouter()
ai = AIState()


# ---------------------------------------------------------------------------
# Gesture action handler — called by camera_stream gesture worker
# ---------------------------------------------------------------------------

# Set of connected voice WebSocket clients that gesture commands are forwarded to
_gesture_ws_clients = set()

def _play_gesture_audio(ai_state, attr: str, fallback: str):
    """Play a preloaded gesture audio array instantly, or fall back to TTS."""
    audio = getattr(ai_state, attr, None)
    if audio is not None:
        try:
            import sounddevice as sd
            sd.stop()
            ai_state.tts._mute_stt()
            sd.play(audio, samplerate=22050)
            sd.wait()
            ai_state.tts._unmute_stt()
            return
        except Exception as e:
            ai_state.tts._unmute_stt()
            logger.warning("Could not play preloaded audio: %s", e)
    ai_state.tts.speak(fallback)


def _handle_gesture(gesture_name: str):
    """
    Called from the gesture worker thread when a gesture fires.
    Dispatches to the appropriate action via the event loop.
    """
    import camera_stream
    loop = getattr(_handle_gesture, "_loop", None)
    if loop is None or not loop.is_running():
        return
    asyncio.run_coroutine_threadsafe(_dispatch_gesture(gesture_name), loop)


async def _dispatch_gesture(gesture_name: str):
    import camera_stream
    logger.info("[gesture_action] Dispatching: %s", gesture_name)

    if gesture_name == camera_stream.GESTURE_OPEN_PALM:
        # Open palm: play greeting only — no queue manipulation
        _play_gesture_audio(ai, "_greeting_audio_array", "Good day, sir.")

    elif gesture_name == camera_stream.GESTURE_THUMBS_UP:
        # Thumbs up: start listening for voice input
        logger.info("[gesture_action] Starting voice listening")
        _play_gesture_audio(ai, "_thumbsup_audio_array", "Yes, sir?")
        for ws in list(_gesture_ws_clients):
            try:
                await ws.send_json({"type": "gesture_command", "command": "start_vosk"})
            except Exception:
                pass

    elif gesture_name == camera_stream.GESTURE_PEACE:
        # Peace sign: submit voice input to AI — no skip/abort here
        logger.info("[gesture_action] Submitting voice input")
        for ws in list(_gesture_ws_clients):
            try:
                await ws.send_json({"type": "gesture_command", "command": "submit_vosk"})
            except Exception:
                pass

    elif gesture_name == camera_stream.GESTURE_CALL_ME:
        # Call me: toggle mute on/off
        new_mute_state = not ai.tts.muted
        logger.info("[gesture_action] Toggling mute: %s", new_mute_state)
        ai.tts.set_muted(new_mute_state)
        for ws in list(_gesture_ws_clients):
            try:
                await ws.send_json({"type": "mute_state", "muted": new_mute_state})
            except Exception:
                pass


def _register_gesture_callback(loop):
    """Register the gesture callback with camera_stream and store the event loop."""
    try:
        import camera_stream
        _handle_gesture._loop = loop
        camera_stream.set_gesture_action_callback(_handle_gesture)
        logger.info("Gesture action callback registered.")
    except Exception as e:
        logger.warning("Could not register gesture callback: %s", e)


def _preload_greeting(tts: PocketAudio):
    """Pre-synthesise all gesture audio at startup so gestures play instantly."""
    GESTURE_PHRASES = {
        "_greeting_audio_array":  "Good day, sir.",
        "_thumbsup_audio_array":  "Yes, sir?",
        "_peace_audio_array":     "Let me think about that...",
        "_callme_audio_array":    "Understood, sir.",
    }
    for attr, phrase in GESTURE_PHRASES.items():
        try:
            audio = tts._synthesise_to_array(phrase)
            setattr(ai, attr, audio)
            logger.info("Preloaded audio: %s", phrase)
        except Exception as e:
            setattr(ai, attr, None)
            logger.warning("Could not preload '%s': %s", phrase, e)


@router.post("/tts/mute")
async def toggle_mute():
    """Toggle TTS mute state. Returns new state."""
    new_state = not ai.tts.muted
    ai.tts.set_muted(new_state)
    # Notify all connected voice clients of the new mute state
    for ws in list(_gesture_ws_clients):
        try:
            await ws.send_json({"type": "mute_state", "muted": new_state})
        except Exception:
            pass
    return {"muted": new_state}

@router.get("/tts/mute")
async def get_mute_state():
    """Get current TTS mute state."""
    return {"muted": ai.tts.muted}

@router.get("/conversations")
async def list_conversations():
    return ai.conv_manager.list_conversations()

class CreateConversationBody(BaseModel):
    title: Optional[str] = None
    messages: Optional[List[dict]] = None

@router.post("/conversations")
async def create_conversation(body: Optional[CreateConversationBody] = None):
    title = (body.title if body else None) or "New Chat"
    messages = body.messages if body and body.messages is not None else None
    return ai.conv_manager.create_conversation(title=title, messages=messages)

@router.get("/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    conv = ai.conv_manager.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv

@router.patch("/conversations/{conv_id}")
async def rename_conversation(conv_id: str, data: dict):
    new_title = data.get("title")
    if not new_title:
        raise HTTPException(status_code=400, detail="Title is required")
    conv = ai.conv_manager.rename_conversation(conv_id, new_title)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv

@router.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    ai.conv_manager.delete_conversation(conv_id)
    return {"status": "success"}

@router.websocket("/ws/chat/{conv_id}")
async def chat_websocket_endpoint(websocket: WebSocket, conv_id: str):
    await websocket.accept()
    
    conv = ai.conv_manager.get_conversation(conv_id)
    if not conv:
        await websocket.send_json({"type": "error", "message": "Conversation not found"})
        await websocket.close()
        return

    await websocket.send_json({
        "type": "history",
        "messages": [{"role": m["role"], "text": m["content"], "hidden": m.get("hidden", False)} for m in conv["messages"]]
    })

    abort_event = asyncio.Event()
    message_queue = asyncio.Queue()

    async def receive_messages():
        try:
            while True:
                data = await websocket.receive_json()
                await message_queue.put(data)
        except:
            pass

    receive_task = asyncio.create_task(receive_messages())

    try:
        while True:
            data = await message_queue.get()
            
            if data["type"] == "send":
                abort_event.clear()
                user_text = data.get("message", "")
                if not user_text:
                    continue

                conv["messages"].append({"role": "user", "content": user_text, "timestamp": time.time()})

                if len(conv["messages"]) == 1:
                    conv["title"] = user_text[:30] + ("..." if len(user_text) > 30 else "")

                ai.conv_manager.update_conversation(conv_id, conv["messages"])

                try:
                    route = _get_route(user_text)
                    logger.debug("[chat] route: %s", route)

                    await websocket.send_json({"type": "stream_start"})

                    if route == "function_gemma":
                        # Run tool_ai in a subprocess so crashes/OOM don't kill the server
                        loop = asyncio.get_event_loop()
                        tool_call_raw, tool_result = await loop.run_in_executor(
                            None, _run_tool_ai_subprocess, user_text
                        )
                        display_reply = str(tool_result) if tool_result else (
                            "I don't have a function for that one, sir. "
                            "I can help with weather, web search, stock prices, or a network scan."
                        )
                        if not abort_event.is_set():
                            await websocket.send_json({"type": "stream_delta", "text": display_reply})
                            await websocket.send_json({"type": "stream_final", "text": display_reply})
                            if tool_call_raw:
                                conv["messages"].append({"role": "assistant", "content": tool_call_raw, "timestamp": time.time(), "hidden": True})
                            conv["messages"].append({"role": "assistant", "content": display_reply, "timestamp": time.time()})
                            ai.conv_manager.update_conversation(conv_id, conv["messages"])
                    else:
                        # Qwen path: use route to set thinking, not UI toggle
                        thinking_mode = route == "qwen_thinking"
                        full_reply = ""
                        response = await ai.generate_response(conv["messages"], thinking=thinking_mode)

                        for chunk in response:
                            while not message_queue.empty():
                                msg = await message_queue.get()
                                if msg.get("type") == "abort":
                                    abort_event.set()

                            if abort_event.is_set():
                                await websocket.send_json({"type": "stream_aborted"})
                                break

                            delta = chunk['choices'][0]['delta']
                            if 'content' in delta:
                                content = delta['content']
                                full_reply += content
                                display_text = full_reply if thinking_mode else strip_think_for_ui(full_reply)
                                await websocket.send_json({"type": "stream_delta", "text": display_text})

                            await asyncio.sleep(0.01)

                        if not abort_event.is_set():
                            display_text = full_reply if thinking_mode else strip_think_for_ui(full_reply)
                            await websocket.send_json({"type": "stream_final", "text": display_text})
                            conv["messages"].append({"role": "assistant", "content": full_reply, "timestamp": time.time()})
                            ai.conv_manager.update_conversation(conv_id, conv["messages"])
                except Exception as e:
                    logger.exception("Chat send error: %s", e)
                    try:
                        await websocket.send_json({"type": "stream_error", "error": str(e)})
                    except Exception:
                        pass
            
            elif data["type"] == "abort":
                abort_event.set()

    except WebSocketDisconnect:
        logger.info("Client disconnected from conversation %s", conv_id)
    finally:
        receive_task.cancel()

@router.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    await websocket.accept()
    logger.info("Voice client connected")
    _gesture_ws_clients.add(websocket)

    abort_event = asyncio.Event()
    message_queue = asyncio.Queue()

    async def receive_messages():
        try:
            while True:
                data = await websocket.receive_json()
                await message_queue.put(data)
        except:
            pass

    receive_task = asyncio.create_task(receive_messages())

    try:
        while True:
            data = await message_queue.get()
            command = data.get("type")
            logger.debug("Voice Command Received: %s", command)

            if command == "start_vosk":
                if not ai.is_vosk_recording:
                    ai.is_vosk_recording = True
                    loop = asyncio.get_event_loop()
                    async def vosk_callback(text):
                        try:
                            await websocket.send_json({"type": "vosk_partial", "text": text})
                        except: pass
                    ai.vosk.start_listening(callback=lambda t: asyncio.run_coroutine_threadsafe(vosk_callback(t), loop))
                    await websocket.send_json({"type": "voice_status", "status": "listening"})

            elif command == "stop_vosk":
                if ai.is_vosk_recording:
                    ai.is_vosk_recording = False
                    logger.debug("Stopping Vosk...")
                    text = ai.vosk.stop_listening()
                    logger.debug("Vosk Final Text: %s", text)
                    transcription_only = data.get("transcription_only", False)
                    if text:
                        await websocket.send_json({"type": "vosk_final", "text": text})
                        if transcription_only:
                            await websocket.send_json({"type": "voice_status", "status": "idle"})
                        else:
                            await websocket.send_json({"type": "voice_status", "status": "thinking"})
                            _play_gesture_audio(ai, "_peace_audio_array", "Let me think about that...")
                            await ai.ai_response_and_speak(websocket, text, abort_event, message_queue)
                    else:
                        await websocket.send_json({"type": "voice_status", "status": "idle"})

            elif command == "submit_vosk":
                # Gesture-triggered submit: stop listening and send to AI (same as stop_vosk but gesture-initiated)
                if ai.is_vosk_recording:
                    ai.is_vosk_recording = False
                    logger.debug("Gesture submit: stopping Vosk...")
                    text = ai.vosk.stop_listening()
                    logger.debug("Vosk Final Text (gesture submit): %s", text)
                    if text:
                        await websocket.send_json({"type": "vosk_final", "text": text})
                        await websocket.send_json({"type": "voice_status", "status": "thinking"})
                        _play_gesture_audio(ai, "_peace_audio_array", "Let me think about that...")
                        await ai.ai_response_and_speak(websocket, text, abort_event, message_queue)
                    else:
                        await websocket.send_json({"type": "voice_status", "status": "idle"})
                else:
                    logger.debug("submit_vosk received but not currently recording — ignoring.")

            elif command == "toggle_voice":
                if not ai.is_recording:
                    logger.debug("Starting Whisper Capture...")
                    abort_event.clear()
                    ai.is_recording = True
                    ai.stt.start_capture()
                    await websocket.send_json({"type": "voice_status", "status": "listening"})
                else:
                    ai.is_recording = False
                    logger.debug("Stopping Whisper and Transcribing...")
                    await websocket.send_json({"type": "voice_status", "status": "thinking"})
                    text = ai.stt.stop_and_transcribe()
                    logger.debug("Whisper Transcription: %s", text)
                    transcription_only = data.get("transcription_only", False)
                    if text:
                        await websocket.send_json({"type": "voice_transcription", "text": text})
                        if transcription_only:
                            await websocket.send_json({"type": "voice_status", "status": "idle"})
                        else:
                            _play_gesture_audio(ai, "_peace_audio_array", "Let me think about that...")
                            await ai.ai_response_and_speak(websocket, text, abort_event, message_queue)
                    else:
                        await websocket.send_json({"type": "voice_status", "status": "idle"})

            elif command == "abort":
                logger.info("Global Abort Requested")
                abort_event.set()

            elif command == "task.list":
                try:
                    from task_scheduler import list_jobs
                    jobs = list_jobs()
                    await websocket.send_json({"type": "task_list", "jobs": jobs})
                except Exception as e:
                    logger.warning("[task.list] %s", e)
                    await websocket.send_json({"type": "task_list", "jobs": []})

            elif command == "task.add":
                try:
                    from task_scheduler import add_job
                    name = data.get("name", "").strip() or "Task"
                    description = (data.get("description") or "").strip()
                    schedule = data.get("schedule")
                    payload = data.get("payload") or {}
                    if not schedule:
                        await websocket.send_json({"type": "task_added", "result": False, "error": "Missing schedule"})
                    else:
                        job = add_job(name=name, description=description, schedule=schedule, payload=payload)
                        await websocket.send_json({"type": "task_added", "result": True, "job": job})
                except Exception as e:
                    logger.warning("[task.add] %s", e)
                    import traceback
                    traceback.print_exc()
                    await websocket.send_json({"type": "task_added", "result": False, "error": str(e)})

            elif command == "task.update":
                try:
                    from task_scheduler import update_job
                    job_id = data.get("id")
                    name = data.get("name", "").strip() or None
                    description = data.get("description")
                    if description is not None:
                        description = (description or "").strip()
                    schedule = data.get("schedule")
                    payload = data.get("payload")
                    if not job_id:
                        await websocket.send_json({"type": "task_updated", "result": False, "error": "Missing id"})
                    else:
                        job = update_job(job_id, name=name, description=description, schedule=schedule, payload=payload)
                        if job:
                            await websocket.send_json({"type": "task_updated", "result": True, "job": job})
                        else:
                            await websocket.send_json({"type": "task_updated", "result": False, "error": "Job not found"})
                except Exception as e:
                    logger.warning("[task.update] %s", e)
                    import traceback
                    traceback.print_exc()
                    await websocket.send_json({"type": "task_updated", "result": False, "error": str(e)})

            elif command == "task.remove":
                try:
                    from task_scheduler import remove_job
                    job_id = data.get("id")
                    if job_id:
                        remove_job(job_id)
                    await websocket.send_json({"type": "task_removed"})
                except Exception as e:
                    logger.warning("[task.remove] %s", e)
                    await websocket.send_json({"type": "task_removed"})

    except WebSocketDisconnect:
        logger.info("Voice client disconnected")
    except Exception as e:
        logger.exception("Voice WebSocket error: %s", e)
        import traceback
        traceback.print_exc()
    finally:
        _gesture_ws_clients.discard(websocket)
        receive_task.cancel()


# ---------------------------------------------------------------------------
# Module-level startup: preload greeting + register gesture callback
# This runs once when app.py imports this module.
# ---------------------------------------------------------------------------

def _on_startup():
    import asyncio
    loop = asyncio.get_event_loop()
    _preload_greeting(ai.tts)
    _register_gesture_callback(loop)

# Schedule startup tasks to run once the event loop is running
import asyncio as _asyncio

async def _startup_tasks():
    _preload_greeting(ai.tts)
    _register_gesture_callback(_asyncio.get_running_loop())
    # Register STT engines so TTS can mute them during playback
    ai.tts.register_stt_engine(ai.stt)
    ai.tts.register_stt_engine(ai.vosk)
    logger.info("STT engines registered with TTS for muting.")

# Register as a router startup event so it runs after the server starts
@router.on_event("startup")
async def on_startup():
    await _startup_tasks()
