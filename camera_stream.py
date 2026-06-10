import asyncio
import json
import logging
import multiprocessing
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import glob
import psutil
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Constants ---
STREAM_WIDTH = 640
STREAM_HEIGHT = 384
STREAM_FPS = 30
DETECTION_FRAME_INTERVAL = 3

CAMERA_RESTART_DELAY = 2.0
CAMERA_MAX_CAPTURE_RETRIES = 5


# ---------------------------------------------------------------------------
# Camera stream process
# ---------------------------------------------------------------------------

def run_stream_process(
    stream_queue: multiprocessing.Queue,
    stop_event: multiprocessing.Event,
    detection_enabled: multiprocessing.Value = None,
    detection_queue: multiprocessing.Queue = None,
    width: int = STREAM_WIDTH,
    height: int = STREAM_HEIGHT,
):
    if IS_WINDOWS:
        _run_opencv_stream(stream_queue, stop_event, detection_enabled, detection_queue, width, height)
    else:
        _run_picamera_stream(stream_queue, stop_event, detection_enabled, detection_queue, width, height)


def _run_opencv_stream(stream_queue, stop_event, detection_enabled, detection_queue, width, height):
    """OpenCV webcam capture for Windows."""

    def init_camera():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError(
                "Could not open webcam (VideoCapture index 0). "
                "Check that your camera is connected and not in use."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, STREAM_FPS)
        return cap

    cap = None
    try:
        cap = init_camera()
    except Exception as e:
        logger.error("Camera init failed: %s", e)
        return

    frame_count = 0
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Frame capture failed, retrying...")
                time.sleep(0.1)
                cap.release()
                time.sleep(CAMERA_RESTART_DELAY)
                if stop_event.is_set():
                    return
                try:
                    cap = init_camera()
                except Exception as e:
                    logger.error("Camera restart failed: %s", e)
                    return
                continue

            # OpenCV gives BGR; convert to RGB for consistency
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if frame_rgb.shape[1] != width or frame_rgb.shape[0] != height:
                frame_rgb = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_LINEAR)

            try:
                stream_queue.put_nowait(frame_rgb)
            except Exception:
                pass

            if detection_enabled is not None and detection_queue is not None:
                if detection_enabled.value and frame_count % DETECTION_FRAME_INTERVAL == 0:
                    try:
                        detection_queue.put_nowait(frame_rgb)
                    except Exception:
                        pass

            frame_count += 1

    finally:
        if cap is not None:
            cap.release()


def _run_picamera_stream(stream_queue, stop_event, detection_enabled, detection_queue, width, height):
    """Original Pi-only picamera2 stream."""
    from picamera2 import Picamera2

    def init_camera():
        picam2 = Picamera2()
        main = {"size": (1280, 720), "format": "RGB888"}
        lores = {"size": (width, height), "format": "RGB888"}
        controls = {"FrameRate": STREAM_FPS, "AfMode": 2, "AfRange": 2}
        config = picam2.create_preview_configuration(main=main, lores=lores, controls=controls)
        picam2.configure(config)
        picam2.start()
        return picam2

    def stop_camera_safe(picam2):
        try:
            picam2.stop()
        except Exception:
            pass
        try:
            picam2.close()
        except Exception:
            pass

    picam2 = None
    try:
        picam2 = init_camera()
    except Exception as e:
        logger.error("Camera init failed: %s", e)
        return

    frame_count = 0
    try:
        while not stop_event.is_set():
            frame_data = None
            for attempt in range(CAMERA_MAX_CAPTURE_RETRIES):
                try:
                    frame_data = picam2.capture_array("lores")
                    break
                except Exception as e:
                    if attempt + 1 >= CAMERA_MAX_CAPTURE_RETRIES:
                        logger.error("Device timeout, restarting: %s", e)
                        stop_camera_safe(picam2)
                        picam2 = None
                        time.sleep(CAMERA_RESTART_DELAY)
                        if stop_event.is_set():
                            return
                        try:
                            picam2 = init_camera()
                        except Exception as e2:
                            logger.error("Camera restart failed: %s", e2)
                            return
                        break
                    time.sleep(0.1)

            if frame_data is None:
                continue

            if len(frame_data.shape) == 2:
                frame = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2RGB)
            elif frame_data.shape[2] == 3:
                frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            else:
                frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)

            try:
                stream_queue.put_nowait(frame)
            except Exception:
                pass

            if detection_enabled is not None and detection_queue is not None:
                if detection_enabled.value and frame_count % DETECTION_FRAME_INTERVAL == 0:
                    try:
                        detection_queue.put_nowait(frame)
                    except Exception:
                        pass

            frame_count += 1

    finally:
        if picam2 is not None:
            stop_camera_safe(picam2)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter()

camera_process = None
stop_event = multiprocessing.Event()
stream_queue = multiprocessing.Queue(maxsize=1)
detection_enabled = multiprocessing.Value('b', False)
detection_queue = multiprocessing.Queue(maxsize=1)

_detection_ws_set = set()
_detection_ws_lock = threading.Lock()
_detection_loop = None
_latest_detections = []
_latest_detections_lock = threading.Lock()
_detection_worker_thread = None
_detection_worker_stop = threading.Event()

# Hailo paths (Pi only — ignored on Windows)
DEFAULT_HEF = PROJECT_ROOT / "models" / "yolov11l.hef"
CONFIG_PATH = PROJECT_ROOT / "hailo_od" / "config.json"


# ---------------------------------------------------------------------------
# Detection worker — YOLOv8 on Windows, Hailo on Pi
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Gesture recogniser — MediaPipe Hands
# ---------------------------------------------------------------------------

# Gesture names (used for display label and action dispatch)
GESTURE_OPEN_PALM    = "open_palm"
GESTURE_THUMBS_UP    = "thumbs_up"
GESTURE_FOUR_FINGERS = "four_fingers"
GESTURE_PEACE        = "peace"
GESTURE_CALL_ME      = "call_me"
GESTURE_INDEX_FINGER = "index_finger"
GESTURE_NONE         = None

GESTURE_HOLD_FRAMES   = 15   # ~1 second at ~15 fps detections
GESTURE_COOLDOWN_SEC  = 3.0

# Shared state written by the gesture worker, read by the broadcast helper
_current_gesture      = None          # gesture being held right now (for label)
_current_gesture_lock = threading.Lock()

# Callback set by chat_ai so gestures can trigger actions
_gesture_action_callback = None  # callable(gesture_name: str)


def set_gesture_action_callback(cb):
    global _gesture_action_callback
    _gesture_action_callback = cb


def _classify_gesture(hand_landmarks):
    """
    Classify a single hand into one of our 5 gestures using MediaPipe landmark indices.
    Returns a gesture constant or GESTURE_NONE.
    Landmark indices: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
    """
    lm = hand_landmarks.landmark

    # Tip and base (MCP) indices for each finger
    THUMB_TIP, THUMB_IP   = 4, 3
    INDEX_TIP, INDEX_MCP  = 8, 5
    MIDDLE_TIP, MIDDLE_MCP = 12, 9
    RING_TIP, RING_MCP    = 16, 13
    PINKY_TIP, PINKY_MCP  = 20, 17
    WRIST                 = 0

    def up(tip, mcp):
        return lm[tip].y < lm[mcp].y  # lower y = higher on screen

    def down(tip, mcp):
        return lm[tip].y > lm[mcp].y

    thumb_up_geom   = lm[THUMB_TIP].y < lm[THUMB_IP].y
    index_up        = up(INDEX_TIP, INDEX_MCP)
    middle_up       = up(MIDDLE_TIP, MIDDLE_MCP)
    ring_up         = up(RING_TIP, RING_MCP)
    pinky_up        = up(PINKY_TIP, PINKY_MCP)

    index_down      = down(INDEX_TIP, INDEX_MCP)
    middle_down     = down(MIDDLE_TIP, MIDDLE_MCP)
    ring_down       = down(RING_TIP, RING_MCP)
    pinky_down      = down(PINKY_TIP, PINKY_MCP)
    thumb_down_geom = lm[THUMB_TIP].y > lm[THUMB_IP].y

    # Open palm: all fingers extended
    if index_up and middle_up and ring_up and pinky_up:
        return GESTURE_OPEN_PALM

    # Thumbs up: thumb up, all fingers curled
    if thumb_up_geom and index_down and middle_down and ring_down and pinky_down:
        return GESTURE_THUMBS_UP

    # Peace / V sign: index + middle up, ring + pinky down
    if index_up and middle_up and ring_down and pinky_down:
        return GESTURE_PEACE

    # Index finger only: index up, middle + ring + pinky down
    if index_up and middle_down and ring_down and pinky_down:
        return GESTURE_INDEX_FINGER

    # Call me: pinky + thumb extended, middle three down
    if pinky_up and index_down and middle_down and ring_down:
        return GESTURE_CALL_ME

    # Four fingers: index + middle + ring + pinky up, thumb tucked
    if index_up and middle_up and ring_up and pinky_up and not thumb_up_geom:
        return GESTURE_FOUR_FINGERS

    return GESTURE_NONE


def _run_detection_worker():
    """Gesture detection worker using MediaPipe Hands (compatible with 0.10.x+)."""
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        logger.error("[gesture] mediapipe not installed. Run: pip install mediapipe")
        return

    # Try new Tasks API first (0.10.x+), fall back to legacy solutions API
    hands = None
    use_legacy = False

    try:
        # New API — requires a model file download
        model_path = str(PROJECT_ROOT / "models" / "hand_landmarker.task")
        if not os.path.isfile(model_path):
            logger.info("[gesture] Downloading hand landmarker model...")
            import urllib.request as _ur
            _ur.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                model_path
            )
            logger.info("[gesture] Hand landmarker model downloaded.")

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        hands = mp_vision.HandLandmarker.create_from_options(options)
        logger.info("[gesture] MediaPipe HandLandmarker (new API) ready.")
    except Exception as e:
        logger.warning("[gesture] New API failed (%s), trying legacy solutions API...", e)
        use_legacy = True

    if use_legacy:
        try:
            mp_hands_legacy = mp.solutions.hands
            hands = mp_hands_legacy.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.6,
            )
            logger.info("[gesture] MediaPipe Hands (legacy API) ready.")
        except Exception as e:
            logger.error("[gesture] Both APIs failed: %s", e)
            return

    hold_gesture   = GESTURE_NONE
    hold_count     = 0
    last_fired     = {}
    last_broadcast = None

    def _process_frame(frame):
        """Process a frame and return landmarks or None. Handles both APIs."""
        if use_legacy:
            results = hands.process(frame)
            if results.multi_hand_landmarks:
                return results.multi_hand_landmarks[0]
            return None
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            results  = hands.detect(mp_image)
            if results.hand_landmarks:
                return results.hand_landmarks[0]
            return None

    def _classify_new_api(landmarks):
        """Classify gesture from new API landmark list (NormalizedLandmark objects)."""
        lm = landmarks  # list of NormalizedLandmark

        THUMB_TIP, THUMB_IP    = 4, 3
        INDEX_TIP, INDEX_MCP   = 8, 5
        MIDDLE_TIP, MIDDLE_MCP = 12, 9
        RING_TIP, RING_MCP     = 16, 13
        PINKY_TIP, PINKY_MCP   = 20, 17

        def up(tip, mcp):   return lm[tip].y < lm[mcp].y
        def down(tip, mcp): return lm[tip].y > lm[mcp].y

        thumb_up   = lm[THUMB_TIP].y < lm[THUMB_IP].y
        index_up   = up(INDEX_TIP, INDEX_MCP)
        middle_up  = up(MIDDLE_TIP, MIDDLE_MCP)
        ring_up    = up(RING_TIP, RING_MCP)
        pinky_up   = up(PINKY_TIP, PINKY_MCP)
        index_down  = down(INDEX_TIP, INDEX_MCP)
        middle_down = down(MIDDLE_TIP, MIDDLE_MCP)
        ring_down   = down(RING_TIP, RING_MCP)
        pinky_down  = down(PINKY_TIP, PINKY_MCP)

        if index_up and middle_up and ring_up and pinky_up:
            return GESTURE_OPEN_PALM
        if thumb_up and index_down and middle_down and ring_down and pinky_down:
            return GESTURE_THUMBS_UP
        if index_up and middle_up and ring_down and pinky_down:
            return GESTURE_PEACE
        if index_up and middle_down and ring_down and pinky_down:
            return GESTURE_INDEX_FINGER
        if pinky_up and index_down and middle_down and ring_down:
            return GESTURE_CALL_ME
        # Four fingers: four fingers up, thumb tucked (thumb tip below thumb IP = not up)
        if index_up and middle_up and ring_up and pinky_up and not thumb_up:
            return GESTURE_FOUR_FINGERS
        return GESTURE_NONE

    try:
        while not _detection_worker_stop.is_set():
            if not detection_enabled.value:
                time.sleep(0.3)
                hold_gesture = GESTURE_NONE
                hold_count   = 0
                continue

            try:
                frame = detection_queue.get(timeout=1.0)
            except Exception:
                continue

            try:
                landmarks = _process_frame(frame)
            except Exception as e:
                logger.warning("[gesture] Frame processing error: %s", e)
                continue

            if landmarks is not None:
                gesture = _classify_gesture(landmarks) if use_legacy else _classify_new_api(landmarks)
            else:
                gesture = GESTURE_NONE

            with _current_gesture_lock:
                _current_gesture = gesture

            if gesture != last_broadcast:
                last_broadcast = gesture
                if _detection_loop:
                    _detection_loop.call_soon_threadsafe(_schedule_broadcast)

            if gesture and gesture == hold_gesture:
                hold_count += 1
            else:
                hold_gesture = gesture
                hold_count   = 1 if gesture else 0

            if gesture and hold_count >= GESTURE_HOLD_FRAMES:
                now  = time.time()
                last = last_fired.get(gesture, 0)
                if now - last >= GESTURE_COOLDOWN_SEC:
                    last_fired[gesture] = now
                    hold_count = 0
                    logger.info("[gesture] Fired: %s", gesture)
                    if _gesture_action_callback:
                        try:
                            _gesture_action_callback(gesture)
                        except Exception as e:
                            logger.warning("[gesture] Action callback error: %s", e)
    finally:
        if hands is not None:
            try:
                hands.close()
            except Exception:
                pass

    logger.info("[gesture] worker stopped.")



# ---------------------------------------------------------------------------
# WebSocket broadcast helpers
# ---------------------------------------------------------------------------

async def _broadcast_detections():
    with _current_gesture_lock:
        gesture = _current_gesture
    msg = json.dumps({"type": "gesture", "gesture": gesture})
    with _detection_ws_lock:
        conns = list(_detection_ws_set)
    for ws in conns:
        try:
            await ws.send_text(msg)
        except Exception:
            pass


def _schedule_broadcast():
    if _detection_loop is None:
        return
    asyncio.run_coroutine_threadsafe(_broadcast_detections(), _detection_loop)


# ---------------------------------------------------------------------------
# System stats
# ---------------------------------------------------------------------------

def get_cpu_temp():
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            if 'cpu_thermal' in temps and temps['cpu_thermal']:
                return temps['cpu_thermal'][0].current
            if 'rp1_adc' in temps and temps['rp1_adc']:
                return temps['rp1_adc'][0].current
            for k, v in temps.items():
                if v:
                    return v[0].current
    except Exception:
        pass
    return 0  # Windows doesn't expose temps via psutil


@router.get("/camera/gesture")
async def get_current_gesture():
    """Simple polling endpoint — returns the currently detected gesture."""
    with _current_gesture_lock:
        gesture = _current_gesture
    return {"gesture": gesture}


@router.get("/system/stats")
async def get_stats():
    return {
        "time": time.strftime("%H:%M:%S"),
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "temperature": get_cpu_temp()
    }


# ---------------------------------------------------------------------------
# Camera endpoints
# ---------------------------------------------------------------------------

def _flush_queue(q):
    """Drain all items from a multiprocessing Queue without blocking."""
    try:
        while True:
            q.get_nowait()
    except Exception:
        pass


@router.post("/camera/start")
async def start_camera():
    global camera_process, stop_event
    if camera_process and camera_process.is_alive():
        return {"status": "already_running"}
    _flush_queue(stream_queue)
    _flush_queue(detection_queue)
    stop_event.clear()
    camera_process = multiprocessing.Process(
        target=run_stream_process,
        args=(stream_queue, stop_event, detection_enabled, detection_queue)
    )
    camera_process.start()
    return {"status": "started"}


@router.post("/camera/stop")
async def stop_camera():
    global camera_process, stop_event
    if camera_process:
        stop_event.set()
        proc = camera_process
        camera_process = None
        # Run the blocking join/terminate in a thread so we don't freeze the event loop
        loop = asyncio.get_running_loop()
        def _kill_proc():
            proc.join(timeout=2)
            if proc.is_alive():
                proc.terminate()
        await loop.run_in_executor(None, _kill_proc)
    _flush_queue(stream_queue)
    _flush_queue(detection_queue)
    return {"status": "stopped"}


def generate_frames():
    while True:
        try:
            frame = stream_queue.get(timeout=1.0)
        except Exception:
            if camera_process and not camera_process.is_alive():
                break
            continue

        if frame is None:
            break

        # Pi camera feeds portrait via ribbon cable; laptop webcam is already landscape
        if not IS_WINDOWS:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = cv2.resize(frame, (480, 800), interpolation=cv2.INTER_LINEAR)

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode('.jpg', frame_bgr)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@router.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.post("/camera/capture")
async def capture_image():
    try:
        frame = stream_queue.get(timeout=2.0)

        if not IS_WINDOWS:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = cv2.resize(frame, (480, 800), interpolation=cv2.INTER_LINEAR)

        timestamp = int(time.time())
        filename = f"capture_{timestamp}.jpg"
        save_path = os.path.join("captures", filename)
        os.makedirs("captures", exist_ok=True)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, frame_bgr)
        logger.info("Captured: %s", save_path)
        return {"status": "success", "filename": filename}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Capture failed: {str(e)}"}


@router.post("/camera/detection/start")
async def start_detection():
    global _detection_loop, _detection_worker_thread
    detection_enabled.value = True
    _detection_loop = asyncio.get_running_loop()
    # Signal any existing worker to stop, then spawn a fresh one
    # Don't join() — let the daemon thread die on its own
    _detection_worker_stop.set()
    _detection_worker_stop.clear()
    _detection_worker_thread = threading.Thread(target=_run_detection_worker, daemon=True)
    _detection_worker_thread.start()
    return {"status": "started"}


@router.post("/camera/detection/stop")
async def stop_detection():
    global _detection_worker_thread
    detection_enabled.value = False
    # Signal the worker to stop — don't join() here as it blocks the event loop
    # The thread is a daemon so it will be cleaned up automatically
    _detection_worker_stop.set()
    _detection_worker_thread = None
    with _latest_detections_lock:
        _latest_detections.clear()
    return {"status": "stopped"}


# ---------------------------------------------------------------------------
# Gallery
# ---------------------------------------------------------------------------

@router.get("/gallery/images")
async def list_gallery_images():
    files = glob.glob("captures/*.jpg")
    files.sort(key=os.path.getmtime, reverse=True)
    images = []
    for f in files:
        filename = os.path.basename(f)
        images.append({"filename": filename, "url": f"/captures/{filename}"})
    return {"status": "success", "images": images}


@router.delete("/gallery/images/{filename}")
async def delete_gallery_image(filename: str):
    file_path = os.path.join("captures", filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return {"status": "success"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    return {"status": "error", "message": "File not found"}


# ---------------------------------------------------------------------------
# Detection WebSocket
# ---------------------------------------------------------------------------

@router.websocket("/ws/detections")
async def detection_websocket(websocket: WebSocket):
    await websocket.accept()
    global _detection_loop
    if _detection_loop is None:
        _detection_loop = asyncio.get_running_loop()
    with _detection_ws_lock:
        _detection_ws_set.add(websocket)
    logger.info("Detection WebSocket connected")
    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "ping"}))
            except WebSocketDisconnect:
                break
            except RuntimeError:
                # Socket already closed — exit cleanly
                break
    finally:
        with _detection_ws_lock:
            _detection_ws_set.discard(websocket)
        logger.info("Detection WebSocket disconnected")
