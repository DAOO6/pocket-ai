import React, { useEffect, useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageCircle, Settings, Hand, Image as GalleryIcon, Code, VolumeX, Volume2 } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useWebSocket } from '../contexts/WebSocketContext.jsx';
import { API_BASE_URL } from '../config.js';

const GESTURE_LABELS = {
    open_palm: { label: 'CLEAR',  color: '#00c8ff' },
    thumbs_up: { label: 'LISTEN', color: '#00ff9f' },
    peace:     { label: 'SUBMIT', color: '#ffdd00' },
    call_me:   { label: 'SKIP',   color: '#cc88ff' },
};

// ─── Arc Reactor SVG Component ───────────────────────────────────────────────
function ArcReactor({ voiceStatus, isRecording, isVoskRecording, isMuted }) {
    const isActive   = voiceStatus === 'speaking' || isRecording || isVoskRecording;
    const isListening = isRecording || isVoskRecording;
    const isSpeaking  = voiceStatus === 'speaking';

    const coreColor  = isMuted ? '#ff2222' : isListening ? '#00ff9f' : isSpeaking ? '#ff6600' : '#00c8ff';
    const glowColor  = isMuted ? 'rgba(255,34,34,0.6)' : isListening ? 'rgba(0,255,159,0.6)' : isSpeaking ? 'rgba(255,102,0,0.6)' : 'rgba(0,200,255,0.5)';

    // Generate tick marks for outer ring
    const ticks = Array.from({ length: 72 }, (_, i) => {
        const angle  = (i * 5 * Math.PI) / 180;
        const isMajor = i % 9 === 0;
        const isMid   = i % 3 === 0;
        const r1 = 118;
        const r2 = isMajor ? 106 : isMid ? 110 : 113;
        return {
            x1: 130 + r1 * Math.cos(angle),
            y1: 130 + r1 * Math.sin(angle),
            x2: 130 + r2 * Math.cos(angle),
            y2: 130 + r2 * Math.sin(angle),
            isMajor,
        };
    });

    // Generate hexagonal segments in middle ring
    const segments = Array.from({ length: 12 }, (_, i) => {
        const startAngle = (i * 30 - 8) * (Math.PI / 180);
        const endAngle   = (i * 30 + 8) * (Math.PI / 180);
        const r = 78;
        const dr = 12;
        const x1 = 130 + r * Math.cos(startAngle);
        const y1 = 130 + r * Math.sin(startAngle);
        const x2 = 130 + r * Math.cos(endAngle);
        const y2 = 130 + r * Math.sin(endAngle);
        const x3 = 130 + (r - dr) * Math.cos(endAngle);
        const y3 = 130 + (r - dr) * Math.sin(endAngle);
        const x4 = 130 + (r - dr) * Math.cos(startAngle);
        const y4 = 130 + (r - dr) * Math.sin(startAngle);
        return `M ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2} L ${x3} ${y3} A ${r - dr} ${r - dr} 0 0 0 ${x4} ${y4} Z`;
    });

    return (
        <svg
            viewBox="0 0 260 260"
            className="w-full h-full"
            style={{ filter: `drop-shadow(0 0 20px ${glowColor})` }}
        >
            <defs>
                <radialGradient id="coreGrad" cx="50%" cy="50%" r="50%">
                    <stop offset="0%"   stopColor="#ffffff" stopOpacity="0.95" />
                    <stop offset="40%"  stopColor={coreColor} stopOpacity="0.9" />
                    <stop offset="100%" stopColor={coreColor} stopOpacity="0.2" />
                </radialGradient>
                <radialGradient id="innerGlow" cx="50%" cy="50%" r="50%">
                    <stop offset="0%"   stopColor={coreColor} stopOpacity="0.3" />
                    <stop offset="100%" stopColor={coreColor} stopOpacity="0" />
                </radialGradient>
                <filter id="glow">
                    <feGaussianBlur stdDeviation="3" result="coloredBlur" />
                    <feMerge>
                        <feMergeNode in="coloredBlur" />
                        <feMergeNode in="SourceGraphic" />
                    </feMerge>
                </filter>
            </defs>

            {/* ── Outermost rotating ring with ticks ── */}
            <motion.g
                style={{ transformBox: 'fill-box', transformOrigin: 'center' }}
                animate={{ rotate: 360 }}
                transition={{ duration: 30, repeat: Infinity, ease: 'linear' }}
            >
                <circle cx="130" cy="130" r="120" fill="none" stroke="#1a4a6e" strokeWidth="1" strokeOpacity="0.6" />
                <circle cx="130" cy="130" r="118" fill="none" stroke="#00c8ff" strokeWidth="0.5" strokeOpacity="0.3" />
                {ticks.map((t, i) => (
                    <line
                        key={i}
                        x1={t.x1} y1={t.y1} x2={t.x2} y2={t.y2}
                        stroke={t.isMajor ? '#00c8ff' : '#1a4a6e'}
                        strokeWidth={t.isMajor ? 1.5 : 0.8}
                        strokeOpacity={t.isMajor ? 0.9 : 0.5}
                    />
                ))}
            </motion.g>

            {/* ── Second ring (counter-rotating dashes) ── */}
            <motion.g
                style={{ transformBox: 'fill-box', transformOrigin: 'center' }}
                animate={{ rotate: -360 }}
                transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
            >
                <circle
                    cx="130" cy="130" r="104"
                    fill="none"
                    stroke="#00c8ff"
                    strokeWidth="1.5"
                    strokeDasharray="8 6"
                    strokeOpacity="0.5"
                />
            </motion.g>

            {/* ── Static structural ring ── */}
            <circle cx="130" cy="130" r="95" fill="none" stroke="#1a4a6e" strokeWidth="2" strokeOpacity="0.8" />
            <circle cx="130" cy="130" r="93" fill="none" stroke="#00c8ff" strokeWidth="0.5" strokeOpacity="0.2" />

            {/* ── Segmented middle ring (slowly rotating CW) ── */}
            <motion.g
                style={{ transformBox: 'fill-box', transformOrigin: 'center' }}
                animate={{ rotate: 360 }}
                transition={{ duration: 45, repeat: Infinity, ease: 'linear' }}
            >
                {segments.map((d, i) => (
                    <path
                        key={i}
                        d={d}
                        fill={i % 3 === 0 ? 'rgba(0,200,255,0.15)' : 'rgba(0,200,255,0.06)'}
                        stroke="#00c8ff"
                        strokeWidth="0.8"
                        strokeOpacity="0.6"
                        filter="url(#glow)"
                    />
                ))}
            </motion.g>

            {/* ── Inner ring (fast CCW) ── */}
            <motion.g
                style={{ transformBox: 'fill-box', transformOrigin: 'center' }}
                animate={{ rotate: -360 }}
                transition={{ duration: 12, repeat: Infinity, ease: 'linear' }}
            >
                <circle
                    cx="130" cy="130" r="58"
                    fill="none"
                    stroke="#00c8ff"
                    strokeWidth="2"
                    strokeDasharray="4 12 1 12"
                    strokeOpacity="0.7"
                    filter="url(#glow)"
                />
            </motion.g>

            {/* ── Radar sweep (only when active) ── */}
            <AnimatePresence>
                {isActive && (
                    <motion.g
                        style={{ transformBox: 'fill-box', transformOrigin: 'center' }}
                        animate={{ rotate: 360 }}
                        transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                        initial={{ opacity: 0 }}
                        exit={{ opacity: 0 }}
                    >
                        <defs>
                            <radialGradient id="sweepGrad" cx="50%" cy="50%" r="50%" gradientUnits="userSpaceOnUse">
                                <stop offset="0%" stopColor={coreColor} stopOpacity="0.5" />
                                <stop offset="100%" stopColor={coreColor} stopOpacity="0" />
                            </radialGradient>
                        </defs>
                        <path
                            d={`M 130 130 L ${130 + 90} 130 A 90 90 0 0 1 ${130 + 90 * Math.cos(-Math.PI / 6)} ${130 + 90 * Math.sin(-Math.PI / 6)} Z`}
                            fill={`url(#sweepGrad)`}
                            opacity="0.4"
                        />
                        <line
                            x1="130" y1="130"
                            x2={130 + 90} y2="130"
                            stroke={coreColor}
                            strokeWidth="1.5"
                            strokeOpacity="0.8"
                        />
                    </motion.g>
                )}
            </AnimatePresence>

            {/* ── Glow fill behind core ── */}
            <circle cx="130" cy="130" r="52" fill="url(#innerGlow)" />

            {/* ── Static inner structure ring ── */}
            <circle cx="130" cy="130" r="50" fill="none" stroke="#00c8ff" strokeWidth="1" strokeOpacity="0.4" />

            {/* ── Triangular inner frame (6 lines from center) ── */}
            {Array.from({ length: 6 }, (_, i) => {
                const angle = (i * 60) * Math.PI / 180;
                return (
                    <line
                        key={i}
                        x1="130" y1="130"
                        x2={130 + 44 * Math.cos(angle)}
                        y2={130 + 44 * Math.sin(angle)}
                        stroke="#00c8ff"
                        strokeWidth="0.8"
                        strokeOpacity="0.35"
                    />
                );
            })}

            {/* ── Hexagonal inner ring ── */}
            <polygon
                points={Array.from({ length: 6 }, (_, i) => {
                    const a = (i * 60 - 30) * Math.PI / 180;
                    return `${130 + 36 * Math.cos(a)},${130 + 36 * Math.sin(a)}`;
                }).join(' ')}
                fill="rgba(0,200,255,0.06)"
                stroke="#00c8ff"
                strokeWidth="1"
                strokeOpacity="0.5"
            />

            {/* ── Core glow circle ── */}
            <motion.circle
                cx="130" cy="130" r="22"
                fill="url(#coreGrad)"
                animate={{
                    r: isActive ? [22, 26, 22] : [22, 24, 22],
                    opacity: isActive ? [0.9, 1, 0.9] : [0.7, 0.9, 0.7],
                }}
                transition={{ duration: isListening ? 0.8 : 2, repeat: Infinity, ease: 'easeInOut' }}
                filter="url(#glow)"
            />

            {/* ── Core outer ring ── */}
            <motion.circle
                cx="130" cy="130" r="28"
                fill="none"
                stroke={coreColor}
                strokeWidth="1.5"
                strokeOpacity="0.8"
                animate={{ strokeOpacity: isActive ? [0.8, 1, 0.8] : [0.4, 0.7, 0.4] }}
                transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
                filter="url(#glow)"
            />

            {/* ── Status label inside ring ── */}
            <text
                x="130" y="174"
                textAnchor="middle"
                fill="#00c8ff"
                fontSize="7"
                fontFamily="Orbitron, sans-serif"
                letterSpacing="3"
                opacity="0.7"
            >
                {isMuted ? 'MUTED' : isListening ? 'LISTENING' : isSpeaking ? 'SPEAKING' : 'STANDBY'}
            </text>
        </svg>
    );
}


// ─── Camera HUD Panel ────────────────────────────────────────────────────────
function CameraPanel({ apiBase, currentGesture }) {
    const gestureInfo = currentGesture ? GESTURE_LABELS[currentGesture] : null;

    const GESTURE_GUIDE = [
        { emoji: '✋', label: 'CLEAR',  color: '#00c8ff' },
        { emoji: '👍', label: 'LISTEN', color: '#00ff9f' },
        { emoji: '✌️', label: 'SUBMIT', color: '#ffdd00' },
        { emoji: '🤙', label: 'SKIP',   color: '#cc88ff' },
    ];

    return (
        <motion.div
            initial={{ opacity: 0, x: 60 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 60 }}
            transition={{ duration: 0.4, ease: 'easeOut' }}
            style={{
                width: 260,
                display: 'flex',
                flexDirection: 'column',
                gap: 8,
                flexShrink: 0,
            }}
        >
            {/* Header label */}
            <div style={{
                fontFamily: 'Orbitron, sans-serif',
                fontSize: '0.5rem',
                letterSpacing: '0.2em',
                color: '#00c8ff',
                opacity: 0.7,
                display: 'flex',
                alignItems: 'center',
                gap: 6,
            }}>
                <span className="animate-status-blink" style={{ width: 4, height: 4, borderRadius: '50%', background: '#00c8ff', display: 'inline-block' }} />
                GESTURE CAM
            </div>

            {/* Camera feed */}
            <div style={{
                position: 'relative',
                border: '1px solid rgba(0,200,255,0.5)',
                background: '#000',
                clipPath: 'polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 12px 100%, 0 calc(100% - 12px))',
                boxShadow: '0 0 16px rgba(0,200,255,0.2)',
                overflow: 'hidden',
                aspectRatio: '4/3',
            }}>
                <img
                    src={`${apiBase}/video_feed`}
                    alt="Gesture Camera"
                    style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
                />

                {/* Scan-line overlay */}
                <div style={{
                    position: 'absolute', inset: 0, pointerEvents: 'none',
                    background: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.08) 2px, rgba(0,0,0,0.08) 4px)',
                }} />

                {/* Cyan glow border inset */}
                <div style={{
                    position: 'absolute', inset: 0, pointerEvents: 'none',
                    boxShadow: 'inset 0 0 20px rgba(0,200,255,0.1)',
                }} />

                {/* Corner brackets */}
                {[
                    { top: 6, left: 6, borderWidth: '1px 0 0 1px' },
                    { top: 6, right: 6, borderWidth: '1px 1px 0 0' },
                    { bottom: 6, left: 6, borderWidth: '0 0 1px 1px' },
                    { bottom: 6, right: 6, borderWidth: '0 1px 1px 0' },
                ].map((s, i) => (
                    <div key={i} style={{
                        position: 'absolute', width: 12, height: 12,
                        borderStyle: 'solid', borderColor: '#00c8ff', opacity: 0.8,
                        ...s,
                    }} />
                ))}

                {/* Active gesture label overlay */}
                <AnimatePresence>
                    {gestureInfo && (
                        <motion.div
                            key={currentGesture}
                            initial={{ opacity: 0, y: -6 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0 }}
                            style={{
                                position: 'absolute', top: 8, left: '50%',
                                transform: 'translateX(-50%)',
                                fontFamily: 'Orbitron, sans-serif',
                                fontSize: '0.45rem',
                                letterSpacing: '0.15em',
                                padding: '2px 8px',
                                border: `1px solid ${gestureInfo.color}`,
                                color: gestureInfo.color,
                                background: `${gestureInfo.color}22`,
                                boxShadow: `0 0 10px ${gestureInfo.color}55`,
                                whiteSpace: 'nowrap',
                            }}
                        >
                            {gestureInfo.label}
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>

            {/* Gesture guide */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(4, 1fr)',
                gap: 4,
            }}>
                {GESTURE_GUIDE.map(({ emoji, label, color }) => (
                    <div key={label} style={{
                        display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2,
                        padding: '4px 2px',
                        border: `1px solid ${color}44`,
                        background: `${color}08`,
                        clipPath: 'polygon(0 0, calc(100% - 4px) 0, 100% 4px, 100% 100%, 4px 100%, 0 calc(100% - 4px))',
                    }}>
                        <span style={{ fontSize: '0.9rem', lineHeight: 1 }}>{emoji}</span>
                        <span style={{ fontFamily: 'Orbitron', fontSize: '0.35rem', letterSpacing: '0.05em', color, opacity: 0.9 }}>{label}</span>
                    </div>
                ))}
            </div>
        </motion.div>
    );
}

// ─── HUD Menu Button ─────────────────────────────────────────────────────────
function HudButton({ icon: Icon, label, onClick, color = '#00c8ff', active = false }) {
    return (
        <motion.button
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.96 }}
            onClick={(e) => { onClick(e); e.currentTarget.blur(); }}
            className="pixel-btn flex flex-col items-center justify-center gap-2 w-full h-full"
            style={{
                borderColor: active ? color : color + '88',
                color: active ? '#fff' : color,
                boxShadow: active
                    ? `0 0 24px ${color}88, inset 0 0 20px ${color}22`
                    : `0 0 8px ${color}44`,
                background: active ? `${color}18` : undefined,
            }}
        >
            <Icon size={32} strokeWidth={1.5} />
            <span style={{ fontSize: '0.6rem', letterSpacing: '0.12em', fontFamily: 'Orbitron, sans-serif', fontWeight: 700 }}>
                {label}
            </span>
        </motion.button>
    );
}

// ─── Main HUD Screen ─────────────────────────────────────────────────────────
export default function Home() {
    const navigate = useNavigate();
    const {
        toggleVoice,
        startVosk,
        stopVosk,
        isRecording,
        isVoskRecording,
        voiceStatus,
        voskText,
        voiceStreamText,
        isVoiceStreaming,
    } = useWebSocket();

    const [showBubble, setShowBubble]         = useState(false);
    const bubbleTimeoutRef                     = useRef(null);
    const [gestureActive, setGestureActive]   = useState(false);
    const [currentGesture, setCurrentGesture] = useState(null);
    const cameraStarted                        = useRef(false);
    const pollTimerRef                         = useRef(null);
    const pressTimer                           = useRef(null);
    const [isHoldMode, setIsHoldMode]         = useState(false);
    const [isMuted, setIsMuted]               = useState(false);
    const enterHoldTimer                       = useRef(null);
    const isEnterHeld                          = useRef(false);

    const toggleMute = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE_URL}/tts/mute`, { method: 'POST' });
            const data = await res.json();
            setIsMuted(data.muted);
        } catch (e) { console.error('Mute toggle failed:', e); }
    }, []);

    // ── Mute state sync on mount + every 2s to catch gesture toggles ─────────
    useEffect(() => {
        const syncMute = () => {
            fetch(`${API_BASE_URL}/tts/mute`)
                .then(r => r.json())
                .then(d => setIsMuted(d.muted))
                .catch(() => {});
        };
        syncMute();
        const interval = setInterval(syncMute, 2000);
        return () => clearInterval(interval);
    }, []);

    // ── Stable refs so keydown handler always calls latest fn versions ─────────
    const toggleVoiceRef = useRef(toggleVoice);
    const startVoskRef   = useRef(startVosk);
    const stopVoskRef    = useRef(stopVosk);
    useEffect(() => { toggleVoiceRef.current = toggleVoice; }, [toggleVoice]);
    useEffect(() => { startVoskRef.current   = startVosk;   }, [startVosk]);
    useEffect(() => { stopVoskRef.current    = stopVosk;    }, [stopVosk]);

    // ── Enter key: tap = toggle voice, hold = hold-to-talk ───────────────────
    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.code !== 'Enter' || e.repeat) return;
            if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
            e.preventDefault();
            isEnterHeld.current = false;
            enterHoldTimer.current = setTimeout(() => {
                isEnterHeld.current = true;
                startVoskRef.current();
            }, 400);
        };
        const handleKeyUp = (e) => {
            if (e.code !== 'Enter') return;
            if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
            e.preventDefault();
            if (enterHoldTimer.current) {
                clearTimeout(enterHoldTimer.current);
                enterHoldTimer.current = null;
            }
            if (isEnterHeld.current) {
                stopVoskRef.current();
                isEnterHeld.current = false;
            } else {
                toggleVoiceRef.current();
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        window.addEventListener('keyup', handleKeyUp);
        return () => {
            window.removeEventListener('keydown', handleKeyDown);
            window.removeEventListener('keyup', handleKeyUp);
            if (enterHoldTimer.current) clearTimeout(enterHoldTimer.current);
        };
    }, []); // empty deps — listeners registered once, refs keep fns current

    // ── Gesture polling ───────────────────────────────────────────────────────
    const startPolling = useCallback(() => {
        if (pollTimerRef.current) return;
        const poll = async () => {
            try {
                const res = await fetch(`${API_BASE_URL}/camera/gesture`);
                if (res.ok) {
                    const data = await res.json();
                    setCurrentGesture(data.gesture || null);
                }
            } catch {}
            if (pollTimerRef.current !== null) pollTimerRef.current = setTimeout(poll, 300);
        };
        pollTimerRef.current = setTimeout(poll, 300);
    }, []);

    const stopPolling = useCallback(() => {
        if (pollTimerRef.current) { clearTimeout(pollTimerRef.current); pollTimerRef.current = null; }
        setCurrentGesture(null);
    }, []);

    const startGestureMode = useCallback(async () => {
        try {
            await fetch(`${API_BASE_URL}/camera/start`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
            await new Promise(r => setTimeout(r, 1000));
            await fetch(`${API_BASE_URL}/camera/detection/start`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
            cameraStarted.current = true;
            setGestureActive(true);
            startPolling();
        } catch (e) { console.error('Gesture mode start failed:', e); }
    }, [startPolling]);

    const stopGestureMode = useCallback(async () => {
        setGestureActive(false);
        stopPolling();
        if (cameraStarted.current) {
            await fetch(`${API_BASE_URL}/camera/detection/stop`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }).catch(() => {});
            await fetch(`${API_BASE_URL}/camera/stop`,            { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }).catch(() => {});
            cameraStarted.current = false;
        }
    }, [stopPolling]);

    useEffect(() => { return () => { stopGestureMode(); }; }, []);

    // ── Speech bubble visibility ──────────────────────────────────────────────
    useEffect(() => {
        const active = isVoiceStreaming || voiceStatus === 'speaking';
        if (active) {
            if (bubbleTimeoutRef.current) clearTimeout(bubbleTimeoutRef.current);
            setShowBubble(true);
        } else if (showBubble) {
            bubbleTimeoutRef.current = setTimeout(() => setShowBubble(false), 1500);
        }
        return () => { if (bubbleTimeoutRef.current) clearTimeout(bubbleTimeoutRef.current); };
    }, [isVoiceStreaming, voiceStatus, showBubble]);

    // ── Press-and-hold vs tap ─────────────────────────────────────────────────
    const handleMouseDown = () => {
        setIsHoldMode(false);
        pressTimer.current = setTimeout(() => { setIsHoldMode(true); startVosk(); }, 400);
    };
    const handleMouseUp = () => {
        if (pressTimer.current) { clearTimeout(pressTimer.current); pressTimer.current = null; }
        if (isHoldMode) { stopVosk(); setIsHoldMode(false); } else { toggleVoice(); }
    };
    const handleMouseLeave = () => {
        if (isHoldMode) { stopVosk(); setIsHoldMode(false); }
        if (pressTimer.current) { clearTimeout(pressTimer.current); pressTimer.current = null; }
    };

    const displayVoiceText = voiceStreamText.trim();
    const gestureInfo = currentGesture ? GESTURE_LABELS[currentGesture] : null;

    return (
        <div
            className="relative w-full h-full overflow-hidden flex flex-row items-stretch p-4 gap-4"
            style={{ background: 'radial-gradient(ellipse at center, #051020 0%, #020b18 70%)' }}
        >
            {/* ── Diagonal accent lines — full screen absolute ── */}
            <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ opacity: 0.15, zIndex: 1 }}>
                <line x1="0" y1="0" x2="40%" y2="100%" stroke="#00c8ff" strokeWidth="0.5" />
                <line x1="100%" y1="0" x2="60%" y2="100%" stroke="#00c8ff" strokeWidth="0.5" />
                <line x1="0" y1="30%" x2="15%" y2="0" stroke="#00c8ff" strokeWidth="0.5" />
                <line x1="100%" y1="70%" x2="85%" y2="100%" stroke="#00c8ff" strokeWidth="0.5" />
            </svg>

            {/* ── Background grid ── */}
            <div
                className="absolute inset-0 pointer-events-none"
                style={{
                    backgroundImage: `
                        linear-gradient(rgba(0,200,255,0.04) 1px, transparent 1px),
                        linear-gradient(90deg, rgba(0,200,255,0.04) 1px, transparent 1px)
                    `,
                    backgroundSize: '48px 48px',
                }}
            />
            {/* Main content column — slides left to make room for camera */}
            <motion.div
                layout
                transition={{ duration: 0.4, ease: 'easeInOut' }}
                style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'space-between', minWidth: 0 }}
            >

            {/* ── TOP BAR ── */}
            <div className="relative z-30 w-full flex items-center justify-between">
                {/* Settings */}
                <motion.button
                    whileHover={{ scale: 1.08 }}
                    whileTap={{ scale: 0.94 }}
                    onClick={(e) => { navigate('/settings'); e.currentTarget.blur(); }}
                    className="p-3 flex items-center justify-center"
                    style={{
                        color: '#00c8ff',
                        border: '1px solid rgba(0,200,255,0.4)',
                        background: 'rgba(0,200,255,0.06)',
                        clipPath: 'polygon(0 0, calc(100% - 8px) 0, 100% 8px, 100% 100%, 8px 100%, 0 calc(100% - 8px))',
                        boxShadow: '0 0 10px rgba(0,200,255,0.2)',
                    }}
                    aria-label="Settings"
                >
                    <Settings size={22} strokeWidth={1.5} />
                </motion.button>

                {/* Title */}
                <div className="flex flex-col items-center">
                    <h1
                        style={{
                            fontFamily: 'Orbitron, sans-serif',
                            fontWeight: 900,
                            fontSize: '1.1rem',
                            letterSpacing: '0.3em',
                            color: '#00c8ff',
                            textShadow: '0 0 12px rgba(0,200,255,0.8), 0 0 30px rgba(0,200,255,0.3)',
                        }}
                    >
                        J.A.R.V.I.S
                    </h1>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '2px' }}>
                        <span
                            className="animate-status-blink"
                            style={{ width: 5, height: 5, borderRadius: '50%', background: '#00ff9f', display: 'inline-block', boxShadow: '0 0 6px #00ff9f' }}
                        />
                        <span style={{ fontFamily: 'Share Tech Mono, monospace', fontSize: '0.55rem', letterSpacing: '0.2em', color: '#00ff9f' }}>
                            SYSTEMS ONLINE
                        </span>
                    </div>
                </div>

                {/* Mute button + Gesture status */}
                <div style={{ minWidth: 60 }} className="flex flex-col items-end gap-2">
                    <motion.button
                        whileHover={{ scale: 1.08 }}
                        whileTap={{ scale: 0.94 }}
                        onClick={(e) => { toggleMute(); e.currentTarget.blur(); }}
                        title={isMuted ? 'Unmute Jarvis' : 'Mute Jarvis'}
                        style={{
                            color: isMuted ? '#ff2222' : '#00c8ff',
                            border: `1px solid ${isMuted ? 'rgba(255,34,34,0.6)' : 'rgba(0,200,255,0.4)'}`,
                            background: isMuted ? 'rgba(255,34,34,0.1)' : 'rgba(0,200,255,0.06)',
                            clipPath: 'polygon(0 0, calc(100% - 8px) 0, 100% 8px, 100% 100%, 8px 100%, 0 calc(100% - 8px))',
                            boxShadow: isMuted ? '0 0 12px rgba(255,34,34,0.4)' : '0 0 10px rgba(0,200,255,0.2)',
                            padding: '0.6rem',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            cursor: 'pointer',
                        }}
                    >
                        {isMuted ? <VolumeX size={20} strokeWidth={1.5} /> : <Volume2 size={20} strokeWidth={1.5} />}
                    </motion.button>
                    <AnimatePresence>
                        {gestureActive && (
                            <motion.div
                                initial={{ opacity: 0, x: 10 }}
                                animate={{ opacity: 1, x: 0 }}
                                exit={{ opacity: 0, x: 10 }}
                                className="flex flex-col items-end gap-1"
                            >
                                <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: '0.5rem', fontFamily: 'Orbitron', letterSpacing: '0.1em', color: '#00c8ff' }}>
                                    <span className="animate-status-blink" style={{ width: 4, height: 4, borderRadius: '50%', background: '#00c8ff', display: 'inline-block' }} />
                                    GESTURE
                                </div>
                                <AnimatePresence>
                                    {gestureInfo && (
                                        <motion.div
                                            key={currentGesture}
                                            initial={{ opacity: 0 }}
                                            animate={{ opacity: 1 }}
                                            exit={{ opacity: 0 }}
                                            style={{
                                                fontFamily: 'Orbitron',
                                                fontSize: '0.5rem',
                                                letterSpacing: '0.1em',
                                                padding: '2px 6px',
                                                border: `1px solid ${gestureInfo.color}`,
                                                color: gestureInfo.color,
                                                background: `${gestureInfo.color}18`,
                                                boxShadow: `0 0 8px ${gestureInfo.color}66`,
                                            }}
                                        >
                                            {gestureInfo.label}
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </div>

            {/* ── ARC REACTOR + SPEECH BUBBLE ── */}
            <div className="relative z-20 flex flex-col items-center" style={{ flex: '0 0 auto' }}>
                {/* Outer ambient glow behind reactor */}
                <div
                    style={{
                        position: 'absolute',
                        width: 280, height: 280,
                        borderRadius: '50%',
                        background: 'radial-gradient(circle, rgba(0,200,255,0.08) 0%, transparent 70%)',
                        pointerEvents: 'none',
                    }}
                />

                {/* Arc Reactor — clickable to toggle voice */}
                <div
                    style={{ width: 240, height: 240, cursor: 'pointer', position: 'relative' }}
                    onMouseDown={handleMouseDown}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={handleMouseLeave}
                    onTouchStart={handleMouseDown}
                    onTouchEnd={handleMouseUp}
                >
                    <ArcReactor
                        voiceStatus={voiceStatus}
                        isRecording={isRecording}
                        isVoskRecording={isVoskRecording}
                        isMuted={isMuted}
                    />
                </div>

                {/* Vosk real-time transcription */}
                <AnimatePresence>
                    {isVoskRecording && voskText && (
                        <motion.div
                            initial={{ opacity: 0, y: 8 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: 8 }}
                            style={{
                                position: 'absolute',
                                top: '50%',
                                left: 'calc(50% + 130px)',
                                transform: 'translateY(-50%)',
                                width: 160,
                                background: 'rgba(2,11,24,0.92)',
                                border: '1px solid #00ff9f',
                                padding: '10px 12px',
                                boxShadow: '0 0 16px rgba(0,255,159,0.3)',
                                clipPath: 'polygon(0 0, calc(100% - 8px) 0, 100% 8px, 100% 100%, 0 100%)',
                                zIndex: 50,
                            }}
                        >
                            <div style={{ fontFamily: 'Share Tech Mono', fontSize: '0.7rem', color: '#00ff9f', lineHeight: 1.5 }}>
                                {voskText}
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* AI response bubble */}
                <AnimatePresence>
                    {showBubble && displayVoiceText && (
                        <motion.div
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: 10 }}
                            style={{
                                marginTop: 16,
                                maxWidth: 280,
                                background: 'rgba(2,11,24,0.95)',
                                border: '1px solid #00c8ff',
                                padding: '12px 16px',
                                boxShadow: '0 0 24px rgba(0,200,255,0.25)',
                                clipPath: 'polygon(8px 0, 100% 0, 100% calc(100% - 8px), calc(100% - 8px) 100%, 0 100%, 0 8px)',
                                position: 'relative',
                            }}
                        >
                            {/* Top data label */}
                            <div style={{ fontFamily: 'Orbitron', fontSize: '0.45rem', letterSpacing: '0.15em', color: '#00c8ff', opacity: 0.6, marginBottom: 6 }}>
                                JARVIS OUTPUT
                            </div>
                            <p style={{ fontFamily: 'Share Tech Mono', fontSize: '0.85rem', color: '#a8d8f0', lineHeight: 1.6 }}>
                                {displayVoiceText}
                            </p>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>

            {/* ── MENU GRID ── */}
            <div className="relative z-10 w-full grid grid-cols-2 gap-3" style={{ maxWidth: 480, flex: '0 0 auto' }}>
                <HudButton
                    icon={MessageCircle}
                    label="CHAT"
                    onClick={() => navigate('/chat')}
                    color="#00c8ff"
                />
                <HudButton
                    icon={Hand}
                    label={gestureActive ? 'GESTURE ON' : 'GESTURE'}
                    onClick={() => gestureActive ? stopGestureMode() : startGestureMode()}
                    color="#00c8ff"
                    active={gestureActive}
                />
                <HudButton
                    icon={Code}
                    label="AGENT"
                    onClick={() => navigate('/tasks')}
                    color="#ff6600"
                />
                <HudButton
                    icon={GalleryIcon}
                    label="GALLERY"
                    onClick={() => navigate('/gallery')}
                    color="#cc88ff"
                />
            </div>

            {/* ── Bottom data strip ── */}
            <div
                className="relative z-10 w-full flex justify-between items-center"
                style={{ fontFamily: 'Share Tech Mono', fontSize: '0.55rem', color: 'rgba(0,200,255,0.35)', letterSpacing: '0.1em' }}
            >
                <span>SYS::JARVIS v2.1</span>
                <span>MK VII</span>
                <span>PWR::ARC REACTOR</span>
            </div>
            </motion.div>

            {/* ── Camera panel — slides in from right when gesture active ── */}
            <AnimatePresence>
                {gestureActive && (
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 20 }}>
                        <CameraPanel apiBase={API_BASE_URL} currentGesture={currentGesture} />
                    </div>
                )}
            </AnimatePresence>
        </div>
    );
}
