import React, { createContext, useContext, useEffect, useRef, useState, useCallback } from 'react';
import { API_URL, WS_URL, CHAT_WS_URL } from '../config.js';
import { apiFetch } from '../apiClient.js';

const WebSocketContext = createContext(null);

export function WebSocketProvider({ children }) {
    const [connStatus, setConnStatus] = useState('connecting'); // connected | disconnected | connecting
    const [chatConnStatus, setChatConnStatus] = useState('disconnected');
    const [messages, setMessages] = useState([]); // Chat messages
    const [streamText, setStreamText] = useState('');
    const [streaming, setStreaming] = useState(false);
    const [voiceStreamText, setVoiceStreamText] = useState('');
    const [isVoiceStreaming, setIsVoiceStreaming] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [isVoskRecording, setIsVoskRecording] = useState(false);
    const [voiceStatus, setVoiceStatus] = useState('idle'); // idle | listening | thinking | speaking
    const [voskText, setVoskText] = useState('');
    const [thinking, setThinking] = useState(false);

    // Multi-conversation state
    const [conversations, setConversations] = useState([]);
    const [currentConvId, setCurrentConvId] = useState(null);
    const [lastApiError, setLastApiError] = useState(null);

    // Generic event listeners for other components
    const eventListeners = useRef({});

    const wsRef = useRef(null);
    const reconnectTimer = useRef(null);

    const addEventListener = useCallback((type, callback) => {
        if (!eventListeners.current[type]) {
            eventListeners.current[type] = [];
        }
        eventListeners.current[type].push(callback);

        // Return unsubscribe function
        return () => {
            eventListeners.current[type] = eventListeners.current[type].filter(cb => cb !== callback);
        };
    }, []);

    const handleServerMessage = useCallback((data) => {
        // 1. Dispatch to generic listeners first
        if (eventListeners.current[data.type]) {
            eventListeners.current[data.type].forEach(cb => cb(data));
        }

        // 2. Handle core chat messages locally (or we could move this out too, but keeping it here for simplicity of migration)
        switch (data.type) {
            case 'history': {
                const history = (data.messages || [])
                    .filter((m) => !m.hidden)
                    .map((m) => ({
                        role: m.role,
                        text: m.text,
                    }));
                setMessages(history);
                break;
            }
            case 'stream_start':
                setStreaming(true);
                setStreamText('');
                break;
            case 'stream_delta':
                setStreamText(data.text || '');
                break;
            case 'stream_final':
                setStreaming(false);
                setMessages((prev) => [
                    ...prev,
                    { role: 'assistant', text: data.text || '' },
                ]);
                setStreamText('');
                break;
            case 'stream_error':
                setStreaming(false);
                setMessages((prev) => [
                    ...prev,
                    { role: 'assistant', text: `⚠ Error: ${data.error || 'Unknown error'}` },
                ]);
                setStreamText('');
                break;
            case 'stream_aborted':
                setStreaming(false);
                setStreamText((prevStreamText) => {
                    if (prevStreamText) {
                        setMessages((prev) => [
                            ...prev,
                            { role: 'assistant', text: prevStreamText + '\n[aborted]' },
                        ]);
                    }
                    return '';
                });
                break;
            case 'session_reset':
                setMessages([]);
                setStreamText('');
                setStreaming(false);
                break;
            case 'voice_status':
                setVoiceStatus(data.status);
                setIsRecording(data.status === 'listening');
                if (data.status === 'idle') {
                    setIsVoiceStreaming(false);
                    setVoiceStreamText('');
                }
                break;
            case 'voice_transcription':
                setMessages((prev) => [...prev, { role: 'user', text: data.text }]);
                setVoskText('');
                break;
            case 'vosk_partial':
                setVoskText(data.text || '');
                break;
            case 'vosk_final':
                // Final Vosk result
                setMessages((prev) => [...prev, { role: 'user', text: data.text }]);
                setVoskText('');
                break;
            case 'ai_start':
                setVoiceStreamText('');
                setIsVoiceStreaming(true);
                break;
            case 'ai_delta':
                setVoiceStreamText(data.text || '');
                break;
            case 'ai_final':
                setVoiceStreamText(data.text || '');
                // We keep isVoiceStreaming true while speaking, speaking status comes from voice_status
                break;
            case 'ai_aborted':
                setVoiceStreamText((prev) => prev + ' [aborted]');
                setTimeout(() => setIsVoiceStreaming(false), 2000);
                break;
            case 'gesture_command':
                // Commands fired by backend gesture worker via voice WebSocket
                if (data.command === 'start_vosk') {
                    setIsVoskRecording(true);
                    setVoskText('');
                    // Send start_vosk to the backend via the same voice WS
                    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                        wsRef.current.send(JSON.stringify({ type: 'start_vosk', transcription_only: false }));
                    }
                } else if (data.command === 'stop_vosk') {
                    setIsVoskRecording(false);
                    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                        wsRef.current.send(JSON.stringify({ type: 'stop_vosk' }));
                    }
                } else if (data.command === 'submit_vosk') {
                    // Stop listening and let backend process what was heard
                    setIsVoskRecording(false);
                    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                        wsRef.current.send(JSON.stringify({ type: 'stop_vosk' }));
                    }
                }
                break;
            default:
                break;
        }
    }, []);

    const connect = useCallback(() => {
        if (reconnectTimer.current) {
            clearTimeout(reconnectTimer.current);
            reconnectTimer.current = null;
        }
        if (wsRef.current) {
            wsRef.current.onclose = null;
            wsRef.current.close();
        }

        setConnStatus('connecting');
        const ws = new WebSocket(WS_URL);
        wsRef.current = ws;

        ws.onopen = () => {
            setConnStatus('connected');
            clearTimeout(reconnectTimer.current);
        };

        ws.onclose = () => {
            setConnStatus('disconnected');
            wsRef.current = null;
            reconnectTimer.current = setTimeout(connect, 3000);
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleServerMessage(data);
            } catch (e) {
                console.error("WS Parse error", e);
            }
        };
    }, [handleServerMessage]);

    const fetchConversations = useCallback(async () => {
        try {
            setLastApiError(null);
            const data = await apiFetch('/conversations');
            setConversations(data);
            return data;
        } catch (e) {
            const msg = e?.message || 'Failed to fetch conversations';
            setLastApiError(msg);
            console.error(msg, e);
        }
    }, []);

    const createConversation = useCallback(async () => {
        try {
            setLastApiError(null);
            const data = await apiFetch('/conversations', { method: 'POST' });
            await fetchConversations();
            return data;
        } catch (e) {
            const msg = e?.message || 'Failed to create conversation';
            setLastApiError(msg);
            console.error(msg, e);
        }
    }, [fetchConversations]);

    const deleteConversation = useCallback(async (id) => {
        try {
            setLastApiError(null);
            await apiFetch(`/conversations/${id}`, { method: 'DELETE' });
            await fetchConversations();
            if (currentConvId === id) {
                setCurrentConvId(null);
                setMessages([]);
            }
        } catch (e) {
            const msg = e?.message || 'Failed to delete conversation';
            setLastApiError(msg);
            console.error(msg, e);
        }
    }, [currentConvId, fetchConversations]);

    const renameConversation = useCallback(async (id, newTitle) => {
        try {
            setLastApiError(null);
            await apiFetch(`/conversations/${id}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title: newTitle })
            });
            await fetchConversations();
        } catch (e) {
            const msg = e?.message || 'Failed to rename conversation';
            setLastApiError(msg);
            console.error(msg, e);
        }
    }, [fetchConversations]);

    const chatWsRef = useRef(null);
    const chatReconnectTimer = useRef(null);
    const currentConvIdRef = useRef(currentConvId);
    currentConvIdRef.current = currentConvId;

    const connectChat = useCallback((convId) => {
        if (!convId) {
            setChatConnStatus('disconnected');
            return;
        }
        if (chatWsRef.current) {
            chatWsRef.current.onclose = null;
            chatWsRef.current.close();
            chatWsRef.current = null;
        }

        setChatConnStatus('connecting');
        const ws = new WebSocket(`${CHAT_WS_URL}/${convId}`);
        chatWsRef.current = ws;

        ws.onopen = () => {
            setChatConnStatus('connected');
            if (chatReconnectTimer.current) {
                clearTimeout(chatReconnectTimer.current);
                chatReconnectTimer.current = null;
            }
        };
        ws.onclose = () => {
            chatWsRef.current = null;
            setChatConnStatus('disconnected');
            chatReconnectTimer.current = setTimeout(() => {
                chatReconnectTimer.current = null;
                if (currentConvIdRef.current === convId) {
                    connectChat(convId);
                }
            }, 2000);
        };
        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleServerMessage(data);
            } catch (e) {
                console.error("Chat WS Parse error", e);
            }
        };
    }, [handleServerMessage]);

    useEffect(() => {
        connect();
        fetchConversations();
        return () => {
            clearTimeout(reconnectTimer.current);
            clearTimeout(chatReconnectTimer.current);
            if (wsRef.current) wsRef.current.close();
            if (chatWsRef.current) chatWsRef.current.close();
        };
    }, [connect, fetchConversations]);

    useEffect(() => {
        if (currentConvId) {
            connectChat(currentConvId);
        }
    }, [currentConvId, connectChat]);

    const sendMessage = useCallback((type, payload = {}) => {
        // task.* commands are only handled by the voice WebSocket (/ws/voice), not chat
        const useVoiceWs = typeof type === 'string' && type.startsWith('task.');
        const targetWs = useVoiceWs ? wsRef.current : (chatWsRef.current?.readyState === WebSocket.OPEN ? chatWsRef.current : wsRef.current);
        if (targetWs && targetWs.readyState === WebSocket.OPEN) {
            targetWs.send(JSON.stringify({ type, ...payload }));
        } else {
            console.warn("WS not connected, cannot send", type);
        }
    }, []);
    const sendVoiceCommand = useCallback((type, payload = {}) => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type, ...payload }));
        } else {
            console.warn("Voice WS not connected, cannot send", type);
        }
    }, []);

    const toggleVoice = useCallback((options = {}) => {
        const { transcriptionOnly = false } = options;
        sendVoiceCommand('toggle_voice', { transcription_only: transcriptionOnly });
    }, [sendVoiceCommand]);

    const startVosk = useCallback((options = {}) => {
        setIsVoskRecording(true);
        setVoskText('');
        const { transcriptionOnly = false } = options;
        sendVoiceCommand('start_vosk', { transcription_only: transcriptionOnly });
    }, [sendVoiceCommand]);

    const stopVosk = useCallback((options = {}) => {
        setIsVoskRecording(false);
        const { transcriptionOnly = false } = options;
        sendVoiceCommand('stop_vosk', { transcription_only: transcriptionOnly });
    }, [sendVoiceCommand]);

    const abort = useCallback(() => {
        if (chatWsRef.current?.readyState === WebSocket.OPEN) {
            chatWsRef.current.send(JSON.stringify({ type: 'abort' }));
        }
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: 'abort' }));
        }
    }, []);

    const toggleThinking = useCallback(() => {
        setThinking(prev => !prev);
    }, []);

    const value = {
        connStatus,
        connect,
        chatConnStatus,
        messages,
        setMessages,
        streamText,
        streaming,
        isRecording: isRecording || isVoskRecording,
        isVoskRecording,
        voiceStatus,
        voskText,
        conversations,
        currentConvId,
        setCurrentConvId,
        fetchConversations,
        createConversation,
        deleteConversation,
        renameConversation,
        sendMessage,
        toggleVoice,
        startVosk,
        stopVosk,
        abort,
        addEventListener,
        voiceStreamText,
        isVoiceStreaming,
        thinking,
        toggleThinking,
        lastApiError,
        clearApiError: () => setLastApiError(null),
    };

    return (
        <WebSocketContext.Provider value={value}>
            {children}
        </WebSocketContext.Provider>
    );
}

export function useWebSocket() {
    return useContext(WebSocketContext);
}
