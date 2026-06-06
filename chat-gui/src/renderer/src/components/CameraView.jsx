import React, { useEffect, useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowLeft, RefreshCw, AlertCircle, Image as GalleryIcon, Camera, Scan } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { API_BASE_URL, WS_BASE_URL } from '../config.js';

// Shared across instances so second mount can cancel first mount's pending stop (React Strict Mode)
const pendingStopTimeoutRef = { current: null };
const STOP_DELAY_MS = 500;

// Stream image size (camera_stream.py).
// On Windows the webcam feed is landscape 640×384 with NO rotation applied.
// On Pi the feed is rotated + resized to portrait 480×800.
// We detect Windows by checking if the stream aspect ratio is landscape.
// To keep this simple we just hardcode Windows dimensions here.
const STREAM_IMAGE_WIDTH = 640;
const STREAM_IMAGE_HEIGHT = 384;
const STREAM_ASPECT = STREAM_IMAGE_WIDTH / STREAM_IMAGE_HEIGHT;

export default function CameraView() {
    const navigate = useNavigate();
    const [status, setStatus] = useState('connecting'); // connecting, connected, error
    const [detections, setDetections] = useState([]);
    const [detectionActive, setDetectionActive] = useState(false);
    const [detectionError, setDetectionError] = useState(null);
    const [flash, setFlash] = useState(false);
    const wsRef = useRef(null);
    const videoContainerRef = useRef(null);
    const [containerSize, setContainerSize] = useState({ width: 1, height: 1 });

    const videoFeedUrl = `${API_BASE_URL}/video_feed`;
    const wsUrl = `${WS_BASE_URL}/ws/detections`;
    const startUrl = `${API_BASE_URL}/camera/start`;
    const stopUrl = `${API_BASE_URL}/camera/stop`;
    const detectionStartUrl = `${API_BASE_URL}/camera/detection/start`;
    const detectionStopUrl = `${API_BASE_URL}/camera/detection/stop`;
    const captureUrl = `${API_BASE_URL}/camera/capture`;

    useEffect(() => {
        let isMounted = true;
        const sessionId = Math.random().toString(36).substring(7); // Simple unique ID

        // Start the camera when the component mounts
        const startCamera = async () => {
            try {
                console.log(`Starting camera session: ${sessionId}`);
                const res = await fetch(startUrl, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sessionId })
                });
                if (res.ok) {
                    if (pendingStopTimeoutRef.current) {
                        clearTimeout(pendingStopTimeoutRef.current);
                        pendingStopTimeoutRef.current = null;
                    }
                    setStatus('connected');
                    console.log("Camera started");
                } else {
                    const err = await res.json().catch(() => ({}));
                    console.error("Camera start failed:", err.message || res.status);
                    if (isMounted) setStatus('error');
                }
            } catch (error) {
                console.error("Failed to start camera:", error);
                if (isMounted) setStatus('error');
            }
        };

        const connectWebSocket = () => {
            console.log('Connecting to WebSocket:', wsUrl);
            const ws = new WebSocket(wsUrl);
            wsRef.current = ws;

            ws.onopen = () => {
                if (isMounted) console.log('WebSocket connected (detections)');
                // Status already 'connected' from camera/start; WS is for detections only
            };

            ws.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    if (message.type === 'detections' && isMounted) {
                        setDetections(Array.isArray(message.data) ? message.data : []);
                    }
                } catch (e) {
                    console.error('Error parsing WebSocket message:', e);
                }
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                if (isMounted) setStatus('error');
            };

            ws.onclose = () => {
                console.log('WebSocket disconnected (detections may stop)');
                // Keep status 'connected' — stream/video are independent of WebSocket
            };
        };

        startCamera().then(() => {
            connectWebSocket();
        });

        // Cleanup: delay stop so React Strict Mode's second mount can send start and cancel this
        return () => {
            isMounted = false;
            if (wsRef.current) {
                wsRef.current.close();
            }

            const sendStop = () => {
                fetch(detectionStopUrl, {
                    method: 'POST',
                    keepalive: true,
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sessionId })
                }).catch(console.error);
                fetch(stopUrl, {
                    method: 'POST',
                    keepalive: true,
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sessionId })
                }).catch(console.error);
            };

            if (pendingStopTimeoutRef.current) {
                clearTimeout(pendingStopTimeoutRef.current);
            }
            pendingStopTimeoutRef.current = setTimeout(() => {
                pendingStopTimeoutRef.current = null;
                sendStop();
            }, STOP_DELAY_MS);
        };
    }, []);

    // Measure video container for object-cover bbox mapping (rotation + crop)
    useEffect(() => {
        const el = videoContainerRef.current;
        if (!el) return;
        const updateSize = () => {
            const { width, height } = el.getBoundingClientRect();
            setContainerSize({ width, height });
        };
        updateSize(); // initial measure
        const ro = new ResizeObserver((entries) => {
            if (!entries.length) return;
            const { width, height } = entries[0].contentRect;
            setContainerSize({ width, height });
        });
        ro.observe(el);
        return () => ro.disconnect();
    }, []);

    const toggleDetection = async () => {
        setDetectionError(null);
        if (detectionActive) {
            // Update UI immediately; backend may block for a few seconds waiting for process
            setDetectionActive(false);
            setDetections([]);
            try {
                await fetch(detectionStopUrl, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: 'default' })
                });
            } catch (e) {
                console.error('Failed to stop detection:', e);
            }
        } else {
            try {
                const res = await fetch(detectionStartUrl, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: 'default' })
                });
                const data = await res.json().catch(() => ({}));
                if (data.status === 'started') {
                    setDetectionActive(true);
                } else {
                    const msg = data.message || data.error || 'Detection failed to start';
                    setDetectionError(msg);
                    console.error('Detection start failed:', msg);
                }
            } catch (e) {
                const msg = e.message || 'Network error';
                setDetectionError(msg);
                console.error('Failed to start detection:', e);
            }
        }
    };

    const captureFrame = async () => {
        // Trigger flash
        setFlash(true);
        setTimeout(() => setFlash(false), 150);

        try {
            const res = await fetch(captureUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: 'default' })
            });
            const data = await res.json();
            if (data.status === 'success') {
                console.log('Image captured:', data.filename);
                // Optional: Show a toast or visual feedback
            } else {
                console.error('Capture failed:', data.message);
            }
        } catch (error) {
            console.error('Error capturing frame:', error);
        }
    };

    // Detection bbox is in original camera frame (640×384) normalized. Video feed is that frame
    // rotated 90° CW then resized to 480×800, and displayed with object-cover (may be cropped).
    const imageNormToContainerNorm = (xImgNorm, yImgNorm) => {
        const { width: cw, height: ch } = containerSize;
        if (cw <= 0 || ch <= 0) return { x: xImgNorm, y: yImgNorm };
        const scale = Math.max(cw / STREAM_IMAGE_WIDTH, ch / STREAM_IMAGE_HEIGHT);
        const displayedW = STREAM_IMAGE_WIDTH * scale;
        const displayedH = STREAM_IMAGE_HEIGHT * scale;
        const offsetLeft = (cw - displayedW) / 2;
        const offsetTop = (ch - displayedH) / 2;
        const xContainer = (offsetLeft + xImgNorm * displayedW) / cw;
        const yContainer = (offsetTop + yImgNorm * displayedH) / ch;
        return { x: xContainer, y: yContainer };
    };

    const renderBoundingBoxes = () => {
        const { width: cw, height: ch } = containerSize;
        if (cw < 10 || ch < 10) return null; // wait for container measure

        return detections.map((det, index) => {
            // bbox is normalized [xmin, ymin, xmax, ymax] in the 640×384 webcam frame.
            // No rotation is applied on Windows so we use the coords directly.
            const [sxMin, syMin, sxMax, syMax] = det.bbox;

            // Map from stream-image normalized to container normalized (object-cover)
            const tl = imageNormToContainerNorm(sxMin, syMin);
            const br = imageNormToContainerNorm(sxMax, syMax);
            const left = Math.max(0, Math.min(1, tl.x));
            const top = Math.max(0, Math.min(1, tl.y));
            const right = Math.max(0, Math.min(1, br.x));
            const bottom = Math.max(0, Math.min(1, br.y));

            const leftPct = `${left * 100}%`;
            const topPct = `${top * 100}%`;
            const widthPct = `${Math.max(0, (right - left) * 100)}%`;
            const heightPct = `${Math.max(0, (bottom - top) * 100)}%`;

            const borderColor = 'rgba(0, 255, 255, 1)';

            return (
                <div
                    key={index}
                    className="absolute border-2 flex flex-col items-start justify-start pointer-events-none"
                    style={{
                        left: leftPct,
                        top: topPct,
                        width: widthPct,
                        height: heightPct,
                        borderColor,
                        boxShadow: '0 0 10px rgba(0,255,255,0.3)'
                    }}
                >
                    <div
                        className="bg-cyan-500 text-black text-[10px] font-bold px-1 py-0.5"
                        style={{ marginTop: '-18px' }}
                    >
                        {det.label} {Math.round(det.confidence * 100)}%
                    </div>
                </div>
            );
        });
    };

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="relative w-full h-full overflow-hidden bg-black"
        >
            {/* Flash Effect */}
            <AnimatePresence>
                {flash && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.05 }}
                        className="absolute inset-0 z-[100] bg-white pointer-events-none"
                    />
                )}
            </AnimatePresence>

            {/* Top Overlay: Back Button */}
            <div className="absolute top-0 left-0 right-0 z-50 p-6 flex justify-between items-start pointer-events-none">
                <button
                    onClick={() => navigate('/')}
                    className="pointer-events-auto pixel-btn p-3 flex items-center justify-center bg-black/50 border-white/50 text-white backdrop-blur-md hover:bg-white hover:text-black hover:border-white transition-all shadow-[0_4px_10px_rgba(0,0,0,0.5)]"
                >
                    <ArrowLeft size={24} />
                </button>

                {/* Status Badge Top Right */}
                <div className="flex flex-col items-end pointer-events-auto">
                    {status === 'connecting' && (
                        <div className="flex items-center gap-2 px-3 py-1 bg-black/60 backdrop-blur border border-white/20 rounded-full text-[10px] text-white font-['Press_Start_2P'] animate-pulse">
                            <RefreshCw size={12} className="animate-spin" />
                            <span>CONNECTING</span>
                        </div>
                    )}
                    {status === 'error' && (
                        <div className="flex items-center gap-2 px-3 py-1 bg-red-500/80 backdrop-blur border border-red-400 rounded-full text-[10px] text-white font-['Press_Start_2P']">
                            <AlertCircle size={12} />
                            <span>OFFLINE</span>
                        </div>
                    )}
                </div>
            </div>

            {/* Main Content: Video & Status */}
            <div ref={videoContainerRef} className="absolute inset-0 z-0 flex items-center justify-center bg-black">
                {/* Video Feed */}
                <img
                    src={videoFeedUrl}
                    className="w-full h-full object-cover"
                    alt="Live Camera Feed"
                    onLoad={() => setStatus((s) => (s === 'connecting' ? 'connected' : s))}
                    onError={(e) => {
                        console.error("Video feed error", e);
                        setStatus('error');
                    }}
                />

                {/* Bounding Boxes */}
                <div className="absolute inset-0 pointer-events-none z-10">
                    {detectionActive && renderBoundingBoxes()}
                </div>

                {/* Error State Overlay (Centered if error) */}
                {status === 'error' && (
                    <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-black/80 backdrop-blur-sm">
                        <div className="p-8 border-4 border-red-500 bg-black flex flex-col items-center gap-4 shadow-[8px_8px_0_rgba(0,0,0,0.5)]">
                            <AlertCircle size={48} className="text-red-500 animate-bounce" />
                            <p className="text-red-500 font-['Press_Start_2P'] text-sm">CAMERA OFFLINE</p>
                            <button
                                onClick={() => window.location.reload()}
                                className="pixel-btn bg-red-500 text-white border-white hover:bg-red-600 px-6 py-3 text-xs"
                            >
                                RETRY
                            </button>
                        </div>
                    </div>
                )}
            </div>

            {/* Bottom Controls Overlay */}
            <div className="absolute bottom-0 left-0 right-0 p-8 pb-10 flex justify-between items-end z-50 bg-gradient-to-t from-black/80 via-black/40 to-transparent h-48 pointer-events-none">
                {/* Gallery Button */}
                <button
                    onClick={() => navigate('/gallery')}
                    className="pointer-events-auto flex flex-col items-center gap-2 group transition-transform active:scale-95"
                >
                    <div className="w-16 h-16 bg-black/50 backdrop-blur border-2 border-white/50 rounded-2xl flex items-center justify-center group-hover:bg-white/20 group-hover:border-white transition-all shadow-[0_4px_10px_rgba(0,0,0,0.3)]">
                        <GalleryIcon size={28} className="text-white drop-shadow-md" />
                    </div>
                    <span className="text-[10px] font-['Press_Start_2P'] text-white/80 drop-shadow-md tracking-wider">GALLERY</span>
                </button>

                {/* Capture Button (Center, Large) */}
                <button
                    onClick={captureFrame}
                    className="pointer-events-auto relative group transition-transform active:scale-95 mx-auto -translate-y-2"
                    aria-label="Capture"
                >
                    <div className="w-24 h-24 rounded-full border-[6px] border-white bg-transparent flex items-center justify-center shadow-[0_0_20px_rgba(0,0,0,0.4)]">
                        <div className="w-20 h-20 rounded-full bg-white group-active:scale-90 transition-transform duration-100 shadow-[inset_0_-4px_8px_rgba(0,0,0,0.2)]" />
                    </div>
                </button>

                {/* Object detection (Hailo) */}
                <button
                    onClick={toggleDetection}
                    title={detectionActive ? 'Stop object detection' : 'Start object detection'}
                    className={`pointer-events-auto flex flex-col items-center gap-2 group transition-transform active:scale-95 ${detectionActive ? 'opacity-100' : 'opacity-80'}`}
                >
                    <div className={`w-16 h-16 rounded-2xl flex items-center justify-center border-2 transition-all shadow-lg ${detectionActive ? 'bg-cyan-500 border-cyan-400 shadow-cyan-500/30' : 'bg-black/50 backdrop-blur border-white/50 group-hover:bg-white/20 group-hover:border-white'}`}>
                        <Scan size={28} className={detectionActive ? 'text-white' : 'text-white drop-shadow-md'} />
                    </div>
                    <span className="text-[10px] font-['Press_Start_2P'] text-white/80 drop-shadow-md tracking-wider">
                        {detectionActive ? 'DETECT ON' : 'DETECT'}
                    </span>
                    {detectionError && <span className="text-[8px] text-red-400 max-w-[80px] truncate">{detectionError}</span>}
                </button>
            </div>
        </motion.div>
    );
}
