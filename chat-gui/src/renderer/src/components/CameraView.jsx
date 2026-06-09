import React, { useEffect, useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowLeft, RefreshCw, AlertCircle, Image as GalleryIcon, Hand } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { API_BASE_URL, WS_BASE_URL } from '../config.js';

const pendingStopTimeoutRef = { current: null };
const STOP_DELAY_MS = 500;

const STREAM_IMAGE_WIDTH = 640;
const STREAM_IMAGE_HEIGHT = 384;
const STREAM_ASPECT = STREAM_IMAGE_WIDTH / STREAM_IMAGE_HEIGHT;

// Human-readable labels and colours for each gesture
const GESTURE_LABELS = {
    open_palm:  { label: '✋ Open Palm',  color: '#00e5ff' },
    thumbs_up:  { label: '👍 Thumbs Up',  color: '#00ff88' },
    fist:       { label: '✊ Fist',        color: '#ff4444' },
    peace:      { label: '✌️ Peace',       color: '#ffdd00' },
    call_me:    { label: '🤙 Call Me',     color: '#cc88ff' },
};

export default function CameraView() {
    const navigate = useNavigate();
    const [status, setStatus] = useState('connecting');
    const [gestureActive, setGestureActive] = useState(false);
    const [currentGesture, setCurrentGesture] = useState(null);
    const [gestureError, setGestureError] = useState(null);
    const [flash, setFlash] = useState(false);
    const wsRef = useRef(null);
    const videoContainerRef = useRef(null);

    const videoFeedUrl    = `${API_BASE_URL}/video_feed`;
    const wsUrl           = `${WS_BASE_URL}/ws/detections`;
    const startUrl        = `${API_BASE_URL}/camera/start`;
    const stopUrl         = `${API_BASE_URL}/camera/stop`;
    const gestureStartUrl = `${API_BASE_URL}/camera/detection/start`;
    const gestureStopUrl  = `${API_BASE_URL}/camera/detection/stop`;
    const captureUrl      = `${API_BASE_URL}/camera/capture`;

    useEffect(() => {
        let isMounted = true;
        const sessionId = Math.random().toString(36).substring(7);

        const startCamera = async () => {
            try {
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
                    if (isMounted) setStatus('connected');
                } else {
                    if (isMounted) setStatus('error');
                }
            } catch {
                if (isMounted) setStatus('error');
            }
        };

        const connectWebSocket = () => {
            const ws = new WebSocket(wsUrl);
            wsRef.current = ws;

            ws.onmessage = (event) => {
                try {
                    const msg = JSON.parse(event.data);
                    if (msg.type === 'gesture' && isMounted) {
                        setCurrentGesture(msg.gesture || null);
                    }
                } catch (e) {
                    console.error('WebSocket parse error:', e);
                }
            };

            ws.onerror = () => { if (isMounted) setStatus('error'); };
            ws.onclose = () => { if (isMounted) setCurrentGesture(null); };
        };

        startCamera().then(connectWebSocket);

        return () => {
            isMounted = false;
            if (wsRef.current) wsRef.current.close();
            if (pendingStopTimeoutRef.current) clearTimeout(pendingStopTimeoutRef.current);
            pendingStopTimeoutRef.current = setTimeout(() => {
                pendingStopTimeoutRef.current = null;
                fetch(gestureStopUrl, { method: 'POST', keepalive: true, headers: { 'Content-Type': 'application/json' }, body: '{}' }).catch(() => {});
                fetch(stopUrl,        { method: 'POST', keepalive: true, headers: { 'Content-Type': 'application/json' }, body: '{}' }).catch(() => {});
            }, STOP_DELAY_MS);
        };
    }, []);

    const toggleGestures = async () => {
        setGestureError(null);
        if (gestureActive) {
            setGestureActive(false);
            setCurrentGesture(null);
            try {
                await fetch(gestureStopUrl, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
            } catch (e) { console.error('Stop gesture failed:', e); }
        } else {
            try {
                const res  = await fetch(gestureStartUrl, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
                const data = await res.json().catch(() => ({}));
                if (data.status === 'started') {
                    setGestureActive(true);
                } else {
                    setGestureError(data.message || 'Failed to start');
                }
            } catch (e) {
                setGestureError(e.message || 'Network error');
            }
        }
    };

    const captureFrame = async () => {
        setFlash(true);
        setTimeout(() => setFlash(false), 150);
        try {
            await fetch(captureUrl, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
        } catch (e) { console.error('Capture failed:', e); }
    };

    const gestureInfo = currentGesture ? GESTURE_LABELS[currentGesture] : null;

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

            {/* Top bar */}
            <div className="absolute top-0 left-0 right-0 z-50 p-6 flex justify-between items-start pointer-events-none">
                <button
                    onClick={() => navigate('/')}
                    className="pointer-events-auto pixel-btn p-3 flex items-center justify-center bg-black/50 border-white/50 text-white backdrop-blur-md hover:bg-white hover:text-black hover:border-white transition-all shadow-[0_4px_10px_rgba(0,0,0,0.5)]"
                >
                    <ArrowLeft size={24} />
                </button>

                <div className="flex flex-col items-end gap-2 pointer-events-none">
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

            {/* Gesture label — shown only when a gesture is detected */}
            <AnimatePresence>
                {gestureActive && gestureInfo && (
                    <motion.div
                        key={currentGesture}
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        transition={{ duration: 0.2 }}
                        className="absolute top-20 left-1/2 -translate-x-1/2 z-50 pointer-events-none"
                    >
                        <div
                            className="px-5 py-2 rounded-full text-sm font-bold backdrop-blur-md border-2 shadow-lg"
                            style={{
                                color: gestureInfo.color,
                                borderColor: gestureInfo.color,
                                backgroundColor: `${gestureInfo.color}22`,
                                boxShadow: `0 0 20px ${gestureInfo.color}55`,
                                fontFamily: "'Press Start 2P', monospace",
                                fontSize: '10px',
                                letterSpacing: '0.05em',
                            }}
                        >
                            {gestureInfo.label}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Video feed */}
            <div ref={videoContainerRef} className="absolute inset-0 z-0 flex items-center justify-center bg-black">
                <img
                    src={videoFeedUrl}
                    className="w-full h-full object-cover"
                    alt="Live Camera Feed"
                    onLoad={() => setStatus(s => s === 'connecting' ? 'connected' : s)}
                    onError={() => setStatus('error')}
                />

                {/* Gesture mode scan-line overlay when active */}
                {gestureActive && (
                    <div
                        className="absolute inset-0 pointer-events-none"
                        style={{
                            background: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,229,255,0.03) 2px, rgba(0,229,255,0.03) 4px)',
                            borderInset: '0 0 0 0',
                        }}
                    />
                )}

                {/* Glowing border when gesture mode is active */}
                {gestureActive && (
                    <div
                        className="absolute inset-0 pointer-events-none rounded-none"
                        style={{ boxShadow: 'inset 0 0 30px rgba(0,229,255,0.15), inset 0 0 2px rgba(0,229,255,0.4)' }}
                    />
                )}

                {/* Error overlay */}
                {status === 'error' && (
                    <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-black/80 backdrop-blur-sm">
                        <div className="p-8 border-4 border-red-500 bg-black flex flex-col items-center gap-4">
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

            {/* Bottom controls */}
            <div className="absolute bottom-0 left-0 right-0 p-8 pb-10 flex justify-between items-end z-50 bg-gradient-to-t from-black/80 via-black/40 to-transparent h-48 pointer-events-none">

                {/* Gallery */}
                <button
                    onClick={() => navigate('/gallery')}
                    className="pointer-events-auto flex flex-col items-center gap-2 group transition-transform active:scale-95"
                >
                    <div className="w-16 h-16 bg-black/50 backdrop-blur border-2 border-white/50 rounded-2xl flex items-center justify-center group-hover:bg-white/20 group-hover:border-white transition-all shadow-lg">
                        <GalleryIcon size={28} className="text-white drop-shadow-md" />
                    </div>
                    <span className="text-[10px] font-['Press_Start_2P'] text-white/80 tracking-wider">GALLERY</span>
                </button>

                {/* Capture */}
                <button
                    onClick={captureFrame}
                    className="pointer-events-auto relative group transition-transform active:scale-95 mx-auto -translate-y-2"
                    aria-label="Capture"
                >
                    <div className="w-24 h-24 rounded-full border-[6px] border-white bg-transparent flex items-center justify-center shadow-[0_0_20px_rgba(0,0,0,0.4)]">
                        <div className="w-20 h-20 rounded-full bg-white group-active:scale-90 transition-transform duration-100 shadow-[inset_0_-4px_8px_rgba(0,0,0,0.2)]" />
                    </div>
                </button>

                {/* Gesture Detection Toggle */}
                <button
                    onClick={toggleGestures}
                    title={gestureActive ? 'Stop gesture detection' : 'Start gesture detection'}
                    className={`pointer-events-auto flex flex-col items-center gap-2 group transition-transform active:scale-95 ${gestureActive ? 'opacity-100' : 'opacity-80'}`}
                >
                    <div
                        className={`w-16 h-16 rounded-2xl flex items-center justify-center border-2 transition-all shadow-lg ${gestureActive ? 'border-cyan-400' : 'bg-black/50 backdrop-blur border-white/50 group-hover:bg-white/20 group-hover:border-white'}`}
                        style={gestureActive ? { backgroundColor: 'rgba(0,229,255,0.15)', boxShadow: '0 0 20px rgba(0,229,255,0.3)' } : {}}
                    >
                        <Hand size={28} className={gestureActive ? 'text-cyan-300' : 'text-white drop-shadow-md'} />
                    </div>
                    <span className="text-[10px] font-['Press_Start_2P'] text-white/80 tracking-wider">
                        {gestureActive ? 'GESTURE ON' : 'GESTURE'}
                    </span>
                    {gestureError && (
                        <span className="text-[8px] text-red-400 max-w-[80px] truncate">{gestureError}</span>
                    )}
                </button>

            </div>

            {/* Gesture guide — shown when gesture mode is active and no gesture detected */}
            <AnimatePresence>
                {gestureActive && !currentGesture && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="absolute bottom-52 left-1/2 -translate-x-1/2 z-40 pointer-events-none"
                    >
                        <div className="flex gap-3 px-4 py-2 bg-black/60 backdrop-blur border border-white/10 rounded-xl">
                            {Object.entries(GESTURE_LABELS).map(([key, val]) => (
                                <div key={key} className="flex flex-col items-center gap-1">
                                    <span className="text-lg">{val.label.split(' ')[0]}</span>
                                </div>
                            ))}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

        </motion.div>
    );
}
