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

def _run_detection_worker():
    if IS_WINDOWS:
        _run_yolo_detection_worker()
    else:
        _run_hailo_detection_worker()


def _run_yolo_detection_worker():
    """
    YOLOv8 nano object detection using Ultralytics.
    Runs entirely on CPU — no special hardware needed.
    Model downloads automatically on first use (~6 MB for nano).
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error(
            "[detection] ultralytics not installed. Run: pip install ultralytics"
        )
        return

    logger.info("[detection] Loading YOLOv8 nano model...")
    try:
        # yolov8n.pt downloads automatically to ~/.ultralytics on first run
        model = YOLO("yolov8n.pt")
        logger.info("[detection] YOLOv8 nano loaded.")
    except Exception as e:
        logger.error("[detection] Failed to load YOLOv8 model: %s", e)
        return

    while not _detection_worker_stop.is_set():
        if not detection_enabled.value:
            time.sleep(0.3)
            continue

        try:
            frame = detection_queue.get(timeout=1.0)
        except Exception:
            continue

        try:
            # frame is RGB numpy array; YOLO accepts RGB directly
            results = model(frame, verbose=False, conf=0.35)
            result = results[0]

            payload = []
            h, w = frame.shape[0], frame.shape[1]

            for box in result.boxes:
                # Normalise bbox to 0-1 range to match the original Hailo output format
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                payload.append({
                    "bbox": [x1 / w, y1 / h, x2 / w, y2 / h],
                    "label": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                })

            with _latest_detections_lock:
                _latest_detections[:] = payload

            if _detection_loop:
                _detection_loop.call_soon_threadsafe(_schedule_broadcast)

        except Exception as e:
            logger.warning("[detection] YOLOv8 inference error: %s", e)

    logger.info("[detection] YOLOv8 worker stopped.")


def _run_hailo_detection_worker():
    """Original Hailo NPU detection worker (Pi only)."""
    import numpy as np

    infer = None
    labels = []
    config_data = {}
    width = height = 640

    try:
        from hailo_od.hailo_inference import HailoInfer
        from hailo_od.toolbox import get_labels, default_preprocess
        from hailo_od.object_detection_post_process import extract_detections
    except Exception as e:
        logger.warning("[detection] hailo_od import failed: %s", e)
        return

    hef_path = str(DEFAULT_HEF)
    if not os.path.isfile(hef_path):
        logger.warning("[detection] HEF not found: %s", hef_path)
        return

    result_holder = []
    done_ev = threading.Event()

    def on_done(completion_info, bindings_list=None):
        if completion_info.exception:
            result_holder.append(("err", None))
        else:
            b = bindings_list[0]
            if len(b._output_names) == 1:
                result_holder.append(("ok", b.output().get_buffer()))
            else:
                result_holder.append(("ok", {
                    n: np.expand_dims(b.output(n).get_buffer(), axis=0)
                    for n in b._output_names
                }))
        done_ev.set()

    while not _detection_worker_stop.is_set():
        if not detection_enabled.value:
            time.sleep(0.3)
            continue

        if infer is None:
            try:
                infer = HailoInfer(hef_path, batch_size=1)
                height, width, _ = infer.get_input_shape()
                labels = get_labels(None)
                config_data = (
                    json.load(open(CONFIG_PATH)) if CONFIG_PATH.exists() else
                    {"visualization_params": {"score_thres": 0.25, "max_boxes_to_draw": 50}}
                )
                logger.info("[detection] Hailo model loaded")
            except Exception as e:
                logger.warning("[detection] model load failed: %s", e)
                time.sleep(1)
                continue

        try:
            frame = detection_queue.get(timeout=1.0)
        except Exception:
            continue

        preprocessed = default_preprocess(frame, width, height)
        result_holder.clear()
        done_ev.clear()

        try:
            infer.run([preprocessed], on_done)
            done_ev.wait(timeout=5.0)
        except Exception as e:
            logger.warning("[detection] inference error: %s", e)
            continue

        if not result_holder or result_holder[0][0] != "ok":
            continue

        raw = result_holder[0][1]

        try:
            if isinstance(raw, dict):
                dets_list = list(raw.values())
            elif hasattr(raw, "shape") and len(raw.shape) >= 2:
                dets_list = _raw_to_per_class_list(raw)
            else:
                dets_list = raw if isinstance(raw, list) else [raw]

            det_dict = extract_detections(frame, dets_list, config_data)
        except Exception as e:
            logger.warning("[detection] postprocess error: %s", e)
            continue

        boxes = det_dict["detection_boxes"]
        classes = det_dict["detection_classes"]
        scores = det_dict["detection_scores"]
        h, w = frame.shape[0], frame.shape[1]

        payload = []
        for i in range(len(boxes)):
            xmin, ymin, xmax, ymax = boxes[i]
            payload.append({
                "bbox": [xmin / w, ymin / h, xmax / w, ymax / h],
                "label": labels[classes[i]] if classes[i] < len(labels) else str(classes[i]),
                "confidence": float(scores[i]),
            })

        with _latest_detections_lock:
            _latest_detections[:] = payload

        if _detection_loop:
            _detection_loop.call_soon_threadsafe(_schedule_broadcast)

    if infer is not None:
        try:
            infer.close()
        except Exception:
            pass
    logger.info("[detection] Hailo worker stopped")


def _raw_to_per_class_list(raw):
    import numpy as np
    raw = np.asarray(raw)
    if raw.size == 0:
        return []
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[-1] >= 6:
        max_cls = int(raw[:, 4].max()) + 1 if raw.shape[0] > 0 else 1
        out = [[] for _ in range(max(80, max_cls))]
        for i in range(raw.shape[0]):
            row = raw[i]
            cid = int(row[4])
            score = float(row[5])
            out[cid].append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), score])
        return [
            np.array(x, dtype=np.float32) if len(x) else np.zeros((0, 5), dtype=np.float32)
            for x in out
        ]
    return [raw]


# ---------------------------------------------------------------------------
# WebSocket broadcast helpers
# ---------------------------------------------------------------------------

async def _broadcast_detections():
    with _latest_detections_lock:
        data = list(_latest_detections)
    msg = json.dumps({"type": "detections", "data": data})
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
