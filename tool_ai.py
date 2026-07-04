"""
Task / tool-running agent: prompt Gemma, parse tool call, run tools (functional),
then create a chat (conversation) with the tool call response.
Uses the same response flow as test_gemma.py (streaming, first_function_call_only).

Windows compatibility: network_scan uses 'arp -a' (built into Windows).
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

logger = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"

_URL_OPENER = urllib.request.build_opener()
_URL_OPENER.addheaders = [("User-Agent", "PocketAI/1.0")]

# Build an SSL context backed by certifi's CA bundle.
# Fixes "SSL: UNEXPECTED_EOF_WHILE_READING" / certificate verify failures on
# Windows, where Python sometimes can't find the system certificate store.
try:
    import ssl
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CONTEXT = None
    logger.warning("[tool_ai] certifi not installed — SSL requests may fail on Windows. Run: pip install certifi")

from config import LOCAL_DIR, TOOLS_PATH, TOOL_REPO_ID, TOOL_FILENAME, TOOL_MODEL_PATH

REPO_ID = TOOL_REPO_ID
FILENAME = TOOL_FILENAME
MODEL_PATH = TOOL_MODEL_PATH

if not os.path.exists(MODEL_PATH):
    logger.info("Model not found at %s. Downloading from Hugging Face...", MODEL_PATH)
    os.makedirs(LOCAL_DIR, exist_ok=True)
    MODEL_PATH = hf_hub_download(
        repo_id=TOOL_REPO_ID,
        filename=TOOL_FILENAME,
        local_dir=LOCAL_DIR,
    )
    logger.info("Download complete!")

PERF = {
    "n_ctx": 2048,
    "n_threads": 4,
    "n_threads_batch": 4,
    "n_batch": 512,
    "n_gpu_layers": -1,
    "use_mmap": True,
    "use_mlock": False,
    "verbose": False,
}

GEN_PERF = {
    "max_tokens": 128,
    "temperature": 0.1,
}


def first_function_call_only(text: str) -> str:
    end_marker = "<end_function_call>"
    idx = text.find(end_marker)
    if idx != -1:
        return text[: idx + len(end_marker)]
    if "<start_function_call>" in text and text.strip().endswith("}"):
        return text.rstrip() + "<end_function_call>"
    return text


def _parse_call_format(payload: str):
    payload = payload.strip()
    if not payload.startswith("call:"):
        return None, None
    rest = payload[5:]
    brace = rest.find("{")
    if brace == -1:
        return None, None
    name = rest[:brace].strip()
    args_str = rest[brace:]
    if not args_str.startswith("{") or "}" not in args_str:
        return None, None
    depth = 0
    end = -1
    for idx, c in enumerate(args_str):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = idx
                break
    if end == -1:
        return None, None
    args_str = args_str[1:end]
    args = {}
    escape = "<escape>"
    pos = 0
    while pos < len(args_str):
        colon = args_str.find(":", pos)
        if colon == -1:
            break
        key = args_str[pos:colon].strip()
        pos = colon + 1
        if pos >= len(args_str):
            break
        if args_str[pos: pos + len(escape)] == escape:
            start_val = pos + len(escape)
            end_val = args_str.find(escape, start_val)
            if end_val == -1:
                break
            value = args_str[start_val:end_val]
            args[key] = value
            pos = end_val + len(escape)
            while pos < len(args_str) and args_str[pos] in " \t,":
                pos += 1
        else:
            end_val = pos
            while end_val < len(args_str) and args_str[end_val] not in ",}":
                end_val += 1
            value = args_str[pos:end_val].strip()
            args[key] = value
            pos = end_val
            while pos < len(args_str) and args_str[pos] in " \t,":
                pos += 1
    return name or None, args if args else {}


def parse_function_call(raw_call: str):
    start_marker = "<start_function_call>"
    end_marker = "<end_function_call>"
    i = raw_call.find(start_marker)
    j = raw_call.find(end_marker)
    if i == -1 or j == -1 or j <= i:
        return None, None
    payload = raw_call[i + len(start_marker): j].strip()
    name, args = _parse_call_format(payload)
    if name:
        return name, args
    for candidate in (payload, _extract_json_object(payload)):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
            name = data.get("name")
            args = data.get("arguments")
            if isinstance(args, str):
                args = json.loads(args) if args.strip() else {}
            if not name:
                continue
            return name, args or {}
        except (json.JSONDecodeError, TypeError):
            continue
    logger.warning("[tool_ai] Could not parse tool call. Raw payload: %r", payload[:500])
    return None, None


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    for i, c in enumerate(text[start:], start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start: i + 1]
    return ""


# --- Tool implementations ---

def _http_get(url: str, timeout: float = 10.0) -> str:
    req = urllib.request.Request(url)
    if _SSL_CONTEXT is not None:
        with _URL_OPENER.open(req, timeout=timeout, context=_SSL_CONTEXT) as resp:
            return resp.read().decode("utf-8", errors="replace")
    with _URL_OPENER.open(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def run_get_weather(arguments: dict) -> str:
    location = (arguments.get("location") or "unknown").strip() or "unknown"
    logger.info("[tool] get_weather(location=%s)", location)
    try:
        geo_url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode(
            {"name": location, "count": 1, "language": "en", "format": "json"}
        )
        geo_data = json.loads(_http_get(geo_url))
        results = geo_data.get("results") or []
        if not results:
            return f"Weather: no location found for '{location}'."
        lat = results[0]["latitude"]
        lon = results[0]["longitude"]
        name = results[0].get("name", location)
        weather_url = (
            "https://api.open-meteo.com/v1/forecast?"
            + urllib.parse.urlencode({
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            })
        )
        weather_data = json.loads(_http_get(weather_url))
        cur = weather_data.get("current", {})
        temp = cur.get("temperature_2m")
        humidity = cur.get("relative_humidity_2m")
        wind = cur.get("wind_speed_10m")
        code = cur.get("weather_code")
        if code == 0:
            conditions = "Clear"
        elif code in (1, 2, 3):
            conditions = "Partly cloudy" if code == 1 else "Cloudy"
        elif code in (45, 48):
            conditions = "Foggy"
        elif 51 <= code <= 67:
            conditions = "Rain/Drizzle"
        elif 71 <= code <= 77:
            conditions = "Snow"
        else:
            conditions = "Precipitation"
        return f"Weather for {name}: {conditions}, {temp}°C, humidity {humidity}%, wind {wind} km/h."
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError) as e:
        return f"Weather for {location}: error — {e}"


def run_activate_security_mode(arguments: dict) -> str:
    logger.info("[tool] activate_security_mode()")
    return "Security mode activated"


def run_web_search(arguments: dict) -> str:
    query = (arguments.get("query") or "").strip()
    logger.info("[tool] web_search(query=%s)", query)
    if not query:
        return "Web search: no query provided."
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=6))
        if not results:
            return f"Web search for '{query}': no results found."
        parts = []
        for r in results:
            title = (r.get("title") or "").strip()
            body = (r.get("body") or "").strip()
            href = (r.get("href") or "").strip()
            line = f"• {title}"
            if body:
                line += f" — {body[:200]}" + ("…" if len(body) > 200 else "")
            if href:
                line += f" ({href})"
            parts.append(line)
        return "Web search:\n" + "\n".join(parts)
    except ImportError:
        pass
    except Exception as e:
        logger.warning("[tool] web_search DDGS error: %s", e)
    try:
        url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode({"q": query, "format": "json"})
        data = json.loads(_http_get(url))
        parts = []
        if data.get("AbstractText"):
            parts.append(data["AbstractText"])
        if data.get("AbstractURL"):
            parts.append(f"Source: {data['AbstractURL']}")
        for t in (data.get("RelatedTopics") or [])[:5]:
            if isinstance(t, dict) and t.get("Text"):
                parts.append(f"• {t['Text']}")
            elif isinstance(t, str):
                parts.append(f"• {t}")
        if not parts:
            return f"Web search for '{query}': no snippets returned. Install ddgs: pip install ddgs"
        return "Web search:\n" + "\n".join(parts)
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
        return f"Web search for '{query}': error — {e}"


def run_network_scan(arguments: dict) -> str:
    """Scan local network via ARP. Works on both Windows and Linux."""
    logger.info("[tool] network_scan()")

    if IS_WINDOWS:
        return _network_scan_windows()
    else:
        return _network_scan_linux()


def _network_scan_windows() -> str:
    """Use 'arp -a' which is available on all Windows versions."""
    try:
        out = subprocess.run(
            ["arp", "-a"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.returncode != 0:
            return f"Network scan failed: {out.stderr.strip()}"

        lines = out.stdout.strip().splitlines()
        devices = []
        for line in lines:
            # arp -a output looks like:
            #   192.168.1.1          aa-bb-cc-dd-ee-ff     dynamic
            # Skip headers/blanks
            line = line.strip()
            if not line or line.startswith("Interface") or line.startswith("Internet"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                ip = parts[0]
                mac = parts[1]
                # Filter out broadcast/multicast entries
                if mac not in ("ff-ff-ff-ff-ff-ff", "ff:ff:ff:ff:ff:ff"):
                    devices.append(f"{ip}  {mac}")

        if not devices:
            return "Network scan: no ARP entries found. (Try pinging your subnet first.)"
        return "Network scan (ARP):\n" + "\n".join(devices)

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        return f"Network scan failed: {e}"


def _network_scan_linux() -> str:
    """Original Linux-based network scan using 'ip neigh' / /proc/net/arp."""
    try:
        out = subprocess.run(
            ["ip", "neigh", "show"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.returncode != 0 and out.stderr:
            return f"Network scan failed: {out.stderr.strip()}"

        lines = [
            l for l in out.stdout.strip().splitlines()
            if l and ("REACHABLE" in l or "STALE" in l or "DELAY" in l)
        ]

        if not lines:
            with open("/proc/net/arp", "r") as f:
                arp_lines = f.read().strip().splitlines()[1:]
            devices = []
            for line in arp_lines:
                parts = line.split()
                if len(parts) >= 6 and parts[3] != "00:00:00:00:00:00":
                    devices.append(f"{parts[0]} {parts[3]}")
            if not devices:
                return "Network scan: no ARP entries found."
            return "Network scan (ARP):\n" + "\n".join(devices)

        devices = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                ip, mac = parts[0], parts[4]
                devices.append(f"{ip} {mac}")
        return "Network scan:\n" + "\n".join(devices) if devices else "Network scan: no devices found."

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        return f"Network scan failed: {e}"


def run_get_stock_price(arguments: dict) -> str:
    symbol = (arguments.get("symbol") or "").strip().upper()
    logger.info("[tool] get_stock_price(symbol=%s)", symbol)
    if not symbol:
        return "Stock: no symbol provided."
    try:
        url = (
            "https://query1.finance.yahoo.com/v8/finance/chart/"
            + urllib.parse.quote(symbol)
            + "?interval=1d&range=1d"
        )
        raw = _http_get(url)
        data = json.loads(raw)
        chart = data.get("chart", {})
        result = (chart.get("result") or [None])[0]
        if not result:
            return f"Stock {symbol}: no data (invalid symbol or no market data)."
        meta = result.get("meta", {})
        price = meta.get("regularMarketPrice") or meta.get("previousClose")
        currency = meta.get("currency", "USD")
        if price is None:
            return f"Stock {symbol}: price not available."
        return f"Stock {symbol}: {currency} {price:.2f} (regular market)."
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TypeError) as e:
        return f"Stock {symbol}: error — {e}"


TOOL_RUNNERS = {
    "get_weather": run_get_weather,
    "activate_security_mode": run_activate_security_mode,
    "web_search": run_web_search,
    "network_scan": run_network_scan,
    "get_stock_price": run_get_stock_price,
}


def run_tool(name: str, arguments: dict) -> str:
    runner = TOOL_RUNNERS.get(name)
    if not runner:
        logger.warning("[tool] unknown tool: %s(%s)", name, arguments)
        # Don't speak the literal "Unknown tool: x" — return None so the
        # caller falls back to a clean in-character message instead.
        return None
    return runner(arguments)


_llm_cache = None
_chat_tools_cache = None


def _get_llm_and_tools():
    global _llm_cache, _chat_tools_cache
    if _llm_cache is not None and _chat_tools_cache is not None:
        return _llm_cache, _chat_tools_cache

    if not os.path.exists(TOOLS_PATH):
        raise FileNotFoundError(f"Tools file not found: {TOOLS_PATH}")

    with open(TOOLS_PATH, "r") as f:
        tools = json.load(f)
    _chat_tools_cache = [{"type": "function", "function": t} for t in tools]

    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        logger.info("Model not found at %s. Downloading...", model_path)
        os.makedirs(LOCAL_DIR, exist_ok=True)
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=LOCAL_DIR)

    logger.info("[tool_ai] Loading FunctionGemma: %s", model_path)
    _llm_cache = Llama(
        model_path=model_path,
        n_ctx=PERF["n_ctx"],
        n_threads=PERF["n_threads"],
        n_threads_batch=PERF["n_threads_batch"],
        n_batch=PERF["n_batch"],
        n_gpu_layers=PERF["n_gpu_layers"],
        use_mmap=PERF["use_mmap"],
        use_mlock=PERF["use_mlock"],
        verbose=PERF["verbose"],
    )
    return _llm_cache, _chat_tools_cache


def preload_tool_model() -> None:
    try:
        _get_llm_and_tools()
        logger.info("[tool_ai] Function Gemma preloaded.")
    except Exception as e:
        logger.warning("[tool_ai] Preload skipped or failed: %s", e)


def run_task_for_backend(prompt: str) -> tuple:
    llm, chat_tools = _get_llm_and_tools()
    return run_task(llm, chat_tools, prompt)


def run_task(llm, chat_tools, user_prompt: str):
    messages = [
        {"role": "developer", "content": "You are a model that can do function calling with the provided functions."},
        {"role": "user", "content": user_prompt},
    ]
    stream = llm.create_chat_completion(
        messages=messages,
        tools=chat_tools,
        max_tokens=GEN_PERF["max_tokens"],
        temperature=GEN_PERF["temperature"],
        stop=["<end_function_call>", "<eos>"],
        stream=True,
    )
    content_parts = []
    for chunk in stream:
        choice = chunk.get("choices", [{}])[0]
        delta = choice.get("delta", {})
        text = delta.get("content") or ""
        if text:
            content_parts.append(text)

    raw = "".join(content_parts)
    tool_call_raw = first_function_call_only(raw)

    if "<start_function_call>" not in tool_call_raw:
        logger.debug("Model did not produce a tool call.")
        return None, None

    name, arguments = parse_function_call(tool_call_raw)
    if not name:
        logger.warning("Could not parse tool call from model output.")
        return tool_call_raw, None

    logger.info("Tool call: %s(%s)", name, arguments)
    result = run_tool(name, arguments)
    return tool_call_raw, result


def create_chat_via_api(title: str, messages: list, api_base: str = "http://127.0.0.1:8000"):
    try:
        payload = json.dumps({"title": title, "messages": messages}).encode("utf-8")
        req = urllib.request.Request(
            f"{api_base}/conversations",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        return data
    except Exception as e:
        logger.warning("Failed to create conversation via API: %s", e)
        return None


def run_backend_mode(prompt: str) -> None:
    try:
        tool_call_raw, tool_result = run_task_for_backend(prompt)
        out = {"tool_call_raw": tool_call_raw, "tool_result": tool_result}
        print(json.dumps(out), flush=True)
    except Exception as e:
        logger.exception("Backend mode error: %s", e)
        out = {"error": str(e), "tool_call_raw": None, "tool_result": None}
        print(json.dumps(out), flush=True)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run Gemma with tools.")
    parser.add_argument("prompt", nargs="?", help="User prompt")
    parser.add_argument("--no-create-chat", action="store_true")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--backend-mode", action="store_true")
    args = parser.parse_args()

    if args.backend_mode:
        prompt = sys.stdin.read().strip()
        if not prompt:
            print(json.dumps({"error": "No prompt on stdin", "tool_call_raw": None, "tool_result": None}), flush=True)
            return 1
        run_backend_mode(prompt)
        return 0

    prompt = args.prompt
    if not prompt:
        prompt = input("Prompt (task message): ").strip()
    if not prompt:
        logger.warning("No prompt provided.")
        return 1

    if not os.path.exists(TOOLS_PATH):
        logger.error("Error: Tools file not found at %s", TOOLS_PATH)
        return 1

    with open(TOOLS_PATH, "r") as f:
        tools = json.load(f)
    chat_tools = [{"type": "function", "function": t} for t in tools]

    logger.info("Loading FunctionGemma model: %s...", MODEL_PATH)
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=PERF["n_ctx"],
        n_threads=PERF["n_threads"],
        n_threads_batch=PERF["n_threads_batch"],
        n_batch=PERF["n_batch"],
        n_gpu_layers=PERF["n_gpu_layers"],
        use_mmap=PERF["use_mmap"],
        use_mlock=PERF["use_mlock"],
        verbose=PERF["verbose"],
    )

    tool_call_raw, tool_result = run_task(llm, chat_tools, prompt)

    chat_messages = [{"role": "user", "content": prompt}]
    if tool_call_raw:
        chat_messages.append({"role": "assistant", "content": tool_call_raw})
    if tool_result is not None:
        chat_messages.append({"role": "assistant", "content": f"[Tool result] {tool_result}"})

    if not args.no_create_chat and chat_messages:
        title = (prompt[:30] + "…") if len(prompt) > 30 else prompt
        conv = create_chat_via_api(title, chat_messages, api_base=args.api_base)
        if conv:
            logger.info("Created conversation: %s — %s", conv.get('id'), conv.get('title', ''))
    else:
        for m in chat_messages:
            logger.info(" %s: %s", m['role'], m['content'][:80])

    return 0


if __name__ == "__main__":
    exit(main())
