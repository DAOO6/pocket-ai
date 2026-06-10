import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageCircle, Settings, Hand, Image as GalleryIcon, Code } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import Avatar from './Avatar';
import { useWebSocket } from '../contexts/WebSocketContext.jsx';
import { useRef, useState, useEffect, useCallback } from 'react';
import { API_BASE_URL } from '../config.js';

const GESTURE_LABELS = {
    open_palm:    { label: '✋ Open Palm',    color: '#00e5ff' },
    thumbs_up:    { label: '👍 Thumbs Up',    color: '#00ff88' },
    peace:        { label: '✌️ Stop + Send',   color: '#ff4444' },
    call_me:      { label: '🤙 Call Me',       color: '#cc88ff' },
};

const MenuButton = ({ icon: Icon, label, onClick, color }) => (
    <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={onClick}
        className="pixel-btn flex flex-col items-center justify-center gap-3 w-full min-h-[100px] flex-1 min-w-0"
        style={{ borderColor: color, color: color }}
    >
        <Icon size={40} />
        <span className="text-sm">{label}</span>
    </motion.button>
);

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
        isVoiceStreaming
    } = useWebSocket();

    const [showBubble, setShowBubble] = useState(false);
    const bubbleTimeoutRef = useRef(null);
    const [gestureActive, setGestureActive] = useState(false);
    const [currentGesture, setCurrentGesture] = useState(null);
    const cameraStarted = useRef(false);
    const pollTimerRef = useRef(null);

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
            if (pollTimerRef.current !== null) {
                pollTimerRef.current = setTimeout(poll, 300);
            }
        };
        pollTimerRef.current = setTimeout(poll, 300);
    }, []);

    const stopPolling = useCallback(() => {
        if (pollTimerRef.current) {
            clearTimeout(pollTimerRef.current);
            pollTimerRef.current = null;
        }
        setCurrentGesture(null);
    }, []);

    const startGestureMode = useCallback(async () => {
        try {
            await fetch(`${API_BASE_URL}/camera/start`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}'
            });
            await new Promise(resolve => setTimeout(resolve, 1000));
            await fetch(`${API_BASE_URL}/camera/detection/start`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}'
            });
            cameraStarted.current = true;
            setGestureActive(true);
            startPolling();
        } catch (e) {
            console.error('Gesture mode start failed:', e);
        }
    }, [startPolling]);

    const stopGestureMode = useCallback(async () => {
        setGestureActive(false);
        stopPolling();
        if (cameraStarted.current) {
            await fetch(`${API_BASE_URL}/camera/detection/stop`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}'
            }).catch(() => {});
            await fetch(`${API_BASE_URL}/camera/stop`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}'
            }).catch(() => {});
            cameraStarted.current = false;
        }
    }, [stopPolling]);

    const toggleGestureMode = () => {
        if (gestureActive) stopGestureMode(); else startGestureMode();
    };

    useEffect(() => {
        return () => { stopGestureMode(); };
    }, []);


    useEffect(() => {
        const active = isVoiceStreaming || voiceStatus === 'speaking';
        if (active) {
            if (bubbleTimeoutRef.current) clearTimeout(bubbleTimeoutRef.current);
            setShowBubble(true);
        } else if (showBubble) {
            bubbleTimeoutRef.current = setTimeout(() => {
                setShowBubble(false);
            }, 1000);
        }
        return () => {
            if (bubbleTimeoutRef.current) clearTimeout(bubbleTimeoutRef.current);
        };
    }, [isVoiceStreaming, voiceStatus, showBubble]);

    const displayVoiceText = voiceStreamText.trim();

    const pressTimer = useRef(null);
    const [isHoldMode, setIsHoldMode] = useState(false);

    const handleMouseDown = () => {
        setIsHoldMode(false);
        pressTimer.current = setTimeout(() => {
            setIsHoldMode(true);
            startVosk();
        }, 400); // 400ms threshold for hold
    };

    const handleMouseUp = () => {
        if (pressTimer.current) {
            clearTimeout(pressTimer.current);
            pressTimer.current = null;
        }

        if (isHoldMode) {
            stopVosk();
            setIsHoldMode(false);
        } else {
            // It was a quick tap
            toggleVoice();
        }
    };

    // Also handle mouse leave to prevent getting stuck in hold mode
    const handleMouseLeave = () => {
        if (isHoldMode) {
            stopVosk();
            setIsHoldMode(false);
        }
        if (pressTimer.current) {
            clearTimeout(pressTimer.current);
            pressTimer.current = null;
        }
    };

    return (
        <div className="relative w-full h-full overflow-hidden bg-[var(--pixel-bg)] flex flex-col items-center justify-center p-4">

            {/* Settings button top left */}
            <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => navigate('/settings')}
                className="absolute top-3 left-3 z-30 p-4 rounded-lg border-2 border-[var(--pixel-border)] bg-[var(--pixel-surface)] shadow-[4px_4px_0_0_rgba(0,0,0,0.5)]"
                style={{ color: '#7dcfff' }}
                aria-label="Settings"
            >
                <Settings size={36} />
            </motion.button>

            {/* Avatar name + status */}
            <div className="flex flex-col items-center gap-2 mb-4 z-10">
                <h1 className="text-2xl font-['Press_Start_2P'] tracking-tight text-[var(--pixel-primary)]">
                    POCKET AI
                </h1>
                <span className="text-xs font-['Press_Start_2P'] tracking-widest text-[var(--pixel-accent)]">
                    SYSTEMS ONLINE
                </span>
            </div>

            {/* Avatar - Centered (higher z when bubble visible so it's not covered by buttons) */}
            <div
                className={`mb-12 relative ${showBubble ? 'z-20' : 'z-10'}`}
                onMouseDown={handleMouseDown}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseLeave}
            >
                <Avatar
                    variant="xl"
                    animate={true}
                    expression={voiceStatus}
                    className={`cursor-pointer transition-all duration-300 ${isRecording ? 'scale-110' : 'hover:scale-105'}`}
                />

                {/* Vosk Real-time Transcription Overlay */}
                <AnimatePresence>
                    {isVoskRecording && voskText && (
                        <motion.div
                            initial={{ opacity: 0, scale: 0.8, y: -10 }}
                            animate={{ opacity: 1, scale: 1, y: 0 }}
                            exit={{ opacity: 0, scale: 0.8 }}
                            className="absolute left-[90px] top-[70px] w-[180px] bg-black/80 backdrop-blur-md p-3 border-2 border-[var(--pixel-accent)] shadow-[4px_4px_0_0_rgba(0,0,0,0.5)] z-50 rounded-lg rounded-tl-none"
                        >
                            <p className="text-[var(--pixel-accent)] text-[10px] leading-relaxed break-words whitespace-pre-wrap max-h-[120px] overflow-y-auto touch-scroll-y">
                                {voskText}
                            </p>
                            {/* Decorative pointer arrow (top left pointing to avatar) */}
                            <div className="absolute left-[-10px] top-[-2px] w-0 h-0 border-r-[10px] border-r-[var(--pixel-accent)] border-b-[10px] border-b-transparent" />
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* AI Response Bubble - below avatar, pointer up toward avatar */}
                <AnimatePresence>
                    {showBubble && displayVoiceText && (
                        <motion.div
                            initial={{ opacity: 0, x: -20, scale: 0.9 }}
                            animate={{ opacity: 1, x: 0, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.9 }}
                            className="absolute left-1/2 top-[150px] -translate-x-1/2 w-[220px] bg-[var(--pixel-surface)] p-4 border-4 border-[var(--pixel-primary)] shadow-[6px_6px_0_0_rgba(0,0,0,0.5)] z-50 rounded-xl"
                        >
                            {/* Pointer arrow at top center pointing up to avatar */}
                            <div className="absolute left-1/2 top-[-14px] -translate-x-1/2 w-0 h-0 border-l-[12px] border-l-transparent border-r-[12px] border-r-transparent border-b-[14px] border-b-[var(--pixel-primary)]" />
                            <p className="text-[var(--pixel-primary)] text-sm font-['VT323'] leading-tight break-words">
                                {displayVoiceText}
                            </p>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>

            {/* Gesture indicator — shown when gesture mode is active */}
            <AnimatePresence>
                {gestureActive && (
                    <motion.div
                        initial={{ opacity: 0, y: -8 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -8 }}
                        className="absolute top-3 right-3 z-30 flex flex-col items-end gap-2 pointer-events-none"
                    >
                        {/* Active pill */}
                        <div className="flex items-center gap-2 px-3 py-1 rounded-full border border-cyan-400/60 bg-cyan-400/10 backdrop-blur">
                            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                            <span className="text-[9px] font-['Press_Start_2P'] text-cyan-400">GESTURE ON</span>
                        </div>
                        {/* Current gesture label */}
                        <AnimatePresence>
                            {currentGesture && GESTURE_LABELS[currentGesture] && (
                                <motion.div
                                    key={currentGesture}
                                    initial={{ opacity: 0, x: 10 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: 10 }}
                                    className="px-3 py-1 rounded-full border-2 backdrop-blur text-[9px] font-['Press_Start_2P']"
                                    style={{
                                        color: GESTURE_LABELS[currentGesture].color,
                                        borderColor: GESTURE_LABELS[currentGesture].color,
                                        backgroundColor: `${GESTURE_LABELS[currentGesture].color}22`,
                                        boxShadow: `0 0 12px ${GESTURE_LABELS[currentGesture].color}44`,
                                    }}
                                >
                                    {GESTURE_LABELS[currentGesture].label}
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Main Menu Grid */}
            <div className="grid grid-cols-2 grid-rows-[1fr_1fr] gap-3 z-10 w-full max-w-[520px] flex-1 min-h-0">
                <MenuButton icon={MessageCircle} label="CHAT" onClick={() => navigate('/chat')} color="var(--pixel-primary)" />
                <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={toggleGestureMode}
                    className="pixel-btn flex flex-col items-center justify-center gap-3 w-full min-h-[100px] flex-1 min-w-0"
                    style={{
                        borderColor: gestureActive ? '#00e5ff' : 'var(--pixel-accent)',
                        color: gestureActive ? '#00e5ff' : 'var(--pixel-accent)',
                        boxShadow: gestureActive ? '0 0 20px rgba(0,229,255,0.3)' : undefined,
                    }}
                >
                    <Hand size={40} />
                    <span className="text-sm">{gestureActive ? 'GESTURE ON' : 'GESTURE'}</span>
                </motion.button>
                <MenuButton icon={Code} label="AGENT" onClick={() => navigate('/tasks')} color="#f7768e" />
                <MenuButton icon={GalleryIcon} label="GALLERY" onClick={() => navigate('/gallery')} color="var(--pixel-secondary)" />
            </div>

            {/* Decorative BG Elements */}
            <div className="absolute inset-0 pointer-events-none opacity-10"
                style={{
                    backgroundImage: 'linear-gradient(var(--pixel-border) 1px, transparent 1px), linear-gradient(90deg, var(--pixel-border) 1px, transparent 1px)',
                    backgroundSize: '40px 40px'
                }}
            />
        </div>
    );
}
