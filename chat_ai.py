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

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from pydantic import BaseModel
from stt_whisper import STTEngine as WhisperEngine
from stt_vosk import STTEngine as VoskEngine
from tts_piper import PocketAudio, split_sentences

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

# --- Model Configuration (from config) ---
from config import (
    CONVERSATIONS_FILE,
    LOCAL_DIR,
    CHAT_REPO_ID as REPO_ID,
    CHAT_FILENAME as FILENAME,
    CHAT_MODEL_PATH as MODEL_PATH,
)

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
                "You are Jarvis, Tony Stark's British AI assistant. "
                "Rules you must always follow: "
                "1. Always address the user as 'sir'. Every single response must include 'sir'. "
                "2. Be concise, calm, and precise. No filler words. "
                "3. Occasionally use dry, understated wit — one clever remark when appropriate. "
                "4. Never break character. You are not a generic AI. You are Jarvis. "
                "5. No bullet points. No markdown. Speak naturally for voice output. "
                "Example: 'Certainly, sir. The weather in London is overcast, 12 degrees. Hardly surprising.'"
            )}
        ]

    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            logger.info("Downloading model...")
            os.makedirs(LOCAL_DIR, exist_ok=True)
            hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=LOCAL_DIR)

        logger.info("Loading LLM...")
        self.llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_threads=4, verbose=False)
        self.stt.load_model()
        self.vosk.load_model()
        logger.info("Chat AI Ready.")

    async def generate_response(self, messages, thinking=True):
        # Prepare messages (skip hidden tool-call entries so the model only sees user/result text)
        llm_messages = []
        for m in messages:
            if m.get("hidden"):
                continue
            content = m["content"]
            # Append mode flag
            if thinking:
                if " /no_think" in content: content = content.replace(" /no_think", " /think")
                elif " /think" not in content: content += " /think"
            else:
                if " /think" in content: content = content.replace(" /think", " /no_think")
                elif " /no_think" not in content: content += " /no_think"
            llm_messages.append({"role": m["role"], "content": content})

        if not any(m["role"] == "system" for m in llm_messages):
            llm_messages.insert(0, {"role": "system", "content": (
                "You are Jarvis, Tony Stark's AI assistant. "
                "Rules you must always follow: "
                "1. Always address the user as 'sir'. Every single response must include 'sir'. "
                "2. Be concise, calm, and precise. No filler words. "
                "3. Occasionally use dry, understated wit — one clever remark when appropriate. "
                "4. Never break character. You are not a generic AI. You are Jarvis. "
                "5. Keep formatting minimal and clean. "
                "Example: 'Certainly, sir. The weather in London is overcast, 12 degrees. Hardly surprising.'"
            )})

        # Sampling params based on thinking mode
        current_temp = 0.6 if thinking else 0.7
        current_top_p = 0.95 if thinking else 0.8

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: self.llm.create_chat_completion(
            messages=llm_messages,
            max_tokens=2048,
            temperature=current_temp,
            top_p=current_top_p,
            top_k=20,
            min_p=0.0,
            presence_penalty=1.5,
            stream=True
        ))
        return response

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
                if tool_call_raw or tool_result is not None:
                    display_text = str(tool_result) if tool_result else "No tool call produced."
                    if tool_call_raw:
                        self.voice_messages.append({"role": "assistant", "content": tool_call_raw, "hidden": True})
                    if tool_result is not None:
                        self.voice_messages.append({"role": "assistant", "content": display_text})
                else:
                    display_text = "No tool call produced."
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
            # Qwen path: stream response and feed TTS sentence-by-sentence
            thinking = route == "qwen_thinking"
            response = await self.generate_response(self.voice_messages, thinking=thinking)

            tts_flushed_len = 0  # index in clean_reply up to which we've flushed to TTS
            speaking_status_sent = False

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

                        # Flush complete sentences to TTS (only non-thinking content; strip unclosed <think> too)
                        clean_reply = strip_think_for_ui(full_response)
                        last_end = max(
                            clean_reply.rfind("."), clean_reply.rfind("!"),
                            clean_reply.rfind("?"), clean_reply.rfind("\n")
                        )
                        if last_end >= tts_flushed_len:
                            to_flush = clean_reply[tts_flushed_len : last_end + 1].strip()
                            tts_flushed_len = last_end + 1
                            for s in split_sentences(to_flush):
                                if not speaking_status_sent:
                                    speaking_status_sent = True
                                    await websocket.send_json({"type": "voice_status", "status": "speaking"})
                                self.tts.enqueue_sentence(s)

                await asyncio.sleep(0.01)

            if not abort_event.is_set():
                logger.debug("AI response complete: %s...", full_response[:50])
                self.voice_messages.append({"role": "assistant", "content": full_response})
                if len(self.voice_messages) > 11:
                    self.voice_messages = [self.voice_messages[0]] + self.voice_messages[-10:]

                clean_reply = strip_think_for_ui(full_response)
                await websocket.send_json({"type": "ai_final", "text": strip_think_for_ui(full_response)})

                # Flush any remaining text to TTS (last sentence or fragment)
                if clean_reply and tts_flushed_len < len(clean_reply):
                    remainder = clean_reply[tts_flushed_len:].strip()
                    if remainder:
                        if not speaking_status_sent:
                            await websocket.send_json({"type": "voice_status", "status": "speaking"})
                        self.tts.enqueue_sentence(remainder)
                if not speaking_status_sent:
                    await websocket.send_json({"type": "voice_status", "status": "idle"})
                # Otherwise "idle" is sent by on_tts_queue_drained when playback finishes

        except Exception as e:
            logger.exception("Error in AI response pipeline: %s", e)
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.send_json({"type": "voice_status", "status": "idle"})

# --- Router Initialization ---
router = APIRouter()
ai = AIState()

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
                        display_reply = str(tool_result) if tool_result is not None else "No tool call produced."
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
                            await ai.ai_response_and_speak(websocket, text, abort_event, message_queue)
                    else:
                        await websocket.send_json({"type": "voice_status", "status": "idle"})

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
        receive_task.cancel()
