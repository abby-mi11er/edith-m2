import { Component, useState, useRef, useEffect, useCallback } from 'react'
import { useStore } from '../store'
import SuggestionsBar from '../components/SuggestionsBar'
import ChatHistory from '../components/ChatHistory'
import { sanitize } from '../security'
import ReactMarkdown from 'react-markdown'
import { apiUrl } from '../api'

/* Error-boundary wrapper — prevents ReactMarkdown crashes from bubbling up */
class SafeMarkdown extends Component<{ children: string }, { crashed: boolean }> {
    state = { crashed: false }
    static getDerivedStateFromError() { return { crashed: true } }
    componentDidCatch(err: Error) { console.warn('[SafeMarkdown] render error:', err.message) }
    componentDidUpdate(prev: { children: string }) {
        if (prev.children !== this.props.children && this.state.crashed) this.setState({ crashed: false })
    }
    render() {
        if (this.state.crashed) return <pre style={{ whiteSpace: 'pre-wrap' }}>{this.props.children}</pre>
        return <ReactMarkdown>{this.props.children}</ReactMarkdown>
    }
}

/* ── Winnie — Research Chat ────────────────────────── */

export default function WinniePanel() {
    const { messages, addMessage, updateLastMessage, isStreaming, setStreaming, committeeMode, setCommitteeMode, clearMessages, setActiveTab, historyOpen, toggleHistory, saveCurrentChat } = useStore()
    const [input, setInput] = useState('')
    const [mode, setMode] = useState<'grounded' | 'lit_review' | 'counterargument' | 'gap_analysis' | 'exam' | 'teaching_intro' | 'office_hours'>('grounded')
    const [modelProvider, setModelProvider] = useState<'gemini' | 'claude'>('gemini')
    const [claudeAvailable, setClaudeAvailable] = useState(false)
    const [followUps, setFollowUps] = useState<string[]>([])
    const bottomRef = useRef<HTMLDivElement>(null)
    const inputRef = useRef<HTMLTextAreaElement>(null)

    useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])
    useEffect(() => { if (!isStreaming) inputRef.current?.focus() }, [isStreaming])

    // Check if Claude is available
    useEffect(() => {
        fetch(apiUrl('/api/connectors/anthropic/status')).then(r => r.json()).then(s => {
            setClaudeAvailable(Boolean(s.available))
        }).catch(() => { })
    }, [])

    // Auto-save chat when streaming completes
    useEffect(() => {
        if (!isStreaming && messages.length > 1) {
            const timer = setTimeout(() => saveCurrentChat(mode), 1000)
            return () => clearTimeout(timer)
        }
    }, [isStreaming])

    const sendMessage = useCallback(async () => {
        const text = sanitize(input)
        if (!text || isStreaming) return

        setInput('')
        setFollowUps([])
        addMessage({ role: 'user', content: text })
        addMessage({ role: 'assistant', content: '' })
        setStreaming(true)

        try {
            if (committeeMode) {
                const res = await fetch(apiUrl('/api/socratic/committee'), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ claim: text, mode }),
                })
                if (!res.ok) throw new Error('Committee request failed')
                const data = await res.json()
                updateLastMessage(data.synthesis || data.answer || data.response || data.result || JSON.stringify(data))
            } else {
                const chatMessages = [
                    ...messages.filter(m => m.content).map(m => ({ role: m.role, content: m.content })),
                    { role: 'user' as const, content: text },
                ]
                const res = await fetch(apiUrl('/chat/stream'), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ messages: chatMessages, mode, model: modelProvider === 'claude' ? 'claude' : undefined }),
                })
                if (!res.ok) throw new Error('Chat request failed')

                const reader = res.body?.getReader()
                if (!reader) throw new Error('No stream reader')

                const decoder = new TextDecoder()
                let fullText = ''
                let sources: any[] = []
                let lineBuf = ''  // Buffer for partial SSE lines split across chunks

                while (true) {
                    const { done, value } = await reader.read()
                    if (done) break

                    const chunk = decoder.decode(value, { stream: true })
                    const lines = (lineBuf + chunk).split('\n')
                    // Last element may be incomplete — buffer it for next iteration
                    lineBuf = lines.pop() || ''

                    for (const line of lines) {
                        if (!line.startsWith('data: ')) continue
                        const payload = line.slice(6).trim()
                        if (payload === '[DONE]') break
                        try {
                            const evt = JSON.parse(payload)
                            // Handle streaming tokens (token-by-token)
                            if (evt.token || evt.text) { fullText += (evt.token || evt.text); updateLastMessage(fullText, sources) }
                            // Handle complete response (sent as single 'done' event with full content)
                            if (evt.content && typeof evt.content === 'string') { fullText = evt.content; updateLastMessage(fullText, sources) }
                            if (evt.sources) sources = evt.sources
                        } catch {
                            // Silently skip malformed JSON — do NOT append raw data to visible text
                        }
                    }
                }
                // Process any remaining buffered line
                if (lineBuf.startsWith('data: ')) {
                    try {
                        const evt = JSON.parse(lineBuf.slice(6).trim())
                        if (evt.content && typeof evt.content === 'string') { fullText = evt.content }
                        if (evt.sources) sources = evt.sources
                    } catch { /* skip */ }
                }
                updateLastMessage(fullText, sources)

                // Follow-up suggestions
                try {
                    const fuRes = await fetch(apiUrl('/api/chat/followups'), {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query: text, response: fullText }),
                    })
                    if (fuRes.ok) {
                        const fuData = await fuRes.json()
                        setFollowUps(fuData.suggestions || fuData.followups || [])
                    }
                } catch { /* non-critical */ }
            }
        } catch (err) {
            updateLastMessage(`Error: ${err instanceof Error ? err.message : 'Connection failed'}`)
        } finally {
            setStreaming(false)
        }
    }, [input, mode, committeeMode, messages, isStreaming, addMessage, updateLastMessage, setStreaming])

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            sendMessage()
        }
    }

    const MODES = [
        { id: 'grounded' as const, label: 'Grounded' },
        { id: 'lit_review' as const, label: 'Lit Review' },
        { id: 'counterargument' as const, label: 'Counter' },
        { id: 'gap_analysis' as const, label: 'Gap Analysis' },
        { id: 'exam' as const, label: 'Exam Prep' },
        { id: 'teaching_intro' as const, label: 'Teach Me' },
        { id: 'office_hours' as const, label: 'Office Hours' },
    ]

    return (
        <div className="panel">
            <div className="winnie-layout">
                <ChatHistory />
                <div className="winnie-layout__main">
                    {/* Mode bar */}
                    <div className="mode-bar">
                        {MODES.map(m => (
                            <button
                                key={m.id}
                                className={`btn btn--sm btn--pill ${mode === m.id ? 'active' : ''}`}
                                onClick={() => setMode(m.id)}
                            >
                                {m.label}
                            </button>
                        ))}
                        <div style={{ flex: 1 }} />
                        <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 'var(--text-sm)', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                            <input type="checkbox" checked={committeeMode} onChange={e => setCommitteeMode(e.target.checked)} />
                            Committee
                        </label>
                        {claudeAvailable && (
                            <select
                                value={modelProvider}
                                onChange={e => setModelProvider(e.target.value as 'gemini' | 'claude')}
                                style={{ fontSize: 'var(--text-sm)', padding: '2px 8px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-primary)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', cursor: 'pointer' }}
                            >
                                <option value="gemini">Gemini</option>
                                <option value="claude">Claude</option>
                            </select>
                        )}
                        <button
                            className={`history-toggle ${historyOpen ? 'history-toggle--active' : ''}`}
                            onClick={toggleHistory}
                            title="Chat history"
                        >
                            🕘 History
                        </button>
                        <button className="btn btn--sm btn--ghost" onClick={clearMessages}>Clear</button>
                    </div>

                    {/* Messages */}
                    <div className="panel__body chat-thread">
                        {messages.length === 0 && (
                            <div className="placeholder">
                                <span className="placeholder__title">Ask Winnie anything</span>
                                <span className="placeholder__text">
                                    Grounded in your research library. Committee mode sends to multiple agents.
                                </span>
                            </div>
                        )}
                        {messages.map((msg, i) => (
                            <div key={i} className={`message message--${msg.role}`}>
                                <div className="message__role">
                                    {msg.role === 'user' ? 'You' : 'E.D.I.T.H.'}
                                </div>
                                <div className="message__bubble">
                                    {msg.role === 'assistant' ? (
                                        isStreaming && i === messages.length - 1 && !msg.content
                                            ? <span style={{ opacity: 0.6 }}>Thinking...</span>
                                            : <SafeMarkdown>{msg.content || ''}</SafeMarkdown>
                                    ) : msg.content}
                                </div>
                                {msg.sources && msg.sources.length > 0 && (
                                    <div className="message__sources">
                                        {msg.sources.map((s, j) => {
                                            const src = typeof s === 'string' ? { title: s } : (s as any)
                                            const meta = src?.meta || src?.metadata || {}
                                            const author = src?.author || meta?.author || ''
                                            const year = src?.year || meta?.year || ''
                                            const title = src?.title || meta?.title || src?.source || src?.name || ''
                                            const page = src?.page || meta?.page
                                            const fname = (src?.file_name || src?.rel_path || meta?.rel_path || meta?.source || src?.source || '').split('/').pop() || ''
                                            const stopwords = new Set(['the', 'of', 'and', 'in', 'on', 'for', 'to', 'a', 'an', 'with', 'university', 'effects', 'place', 'chapter', 'paper', 'voting', 'voter', 'turnout', 'election', 'democracy', 'political', 'american', 'journal', 'review', 'article', 'document', 'codebook', 'abstract', 'introduction', 'methods', 'results', 'discussion', 'appendix', 'notes', 'reading', 'report', 'summary', 'studies', 'research', 'author'])
                                            // Build author-date label: "Acemoglu (2001)" or "Acemoglu (2001) p.5"
                                            let label = ''
                                            // Check if author is a real name (not a stopword)
                                            const authorClean = author && !stopwords.has(author.trim().toLowerCase()) && !stopwords.has(author.trim().split(/\s+/)[0].toLowerCase()) ? author : ''
                                            if (authorClean) {
                                                // Extract last name from "First Last" or "Last, First"
                                                const parts = authorClean.includes(',') ? authorClean.split(',') : authorClean.trim().split(/\s+/)
                                                const lastName = authorClean.includes(',') ? parts[0].trim() : parts[parts.length - 1]
                                                label = year ? `${lastName} (${year})` : lastName
                                                if (page && page > 0) label += ` p.${page}`
                                            } else if (fname) {
                                                // Try extracting author from filename: "Acemoglu_2001.pdf" → "Acemoglu (2001)"
                                                const stem = fname.replace(/\.(pdf|txt|docx|md|tex)$/i, '').replace(/^[\d_\-.\s]+/, '')
                                                const yrMatch = stem.match(/\b((?:19|20)\d{2})\b/)
                                                const beforeYr = yrMatch ? stem.slice(0, yrMatch.index).replace(/[_-]/g, ' ').trim() : ''
                                                const firstWord = beforeYr.split(/\s+/)[0] || ''
                                                if (firstWord && firstWord[0] === firstWord[0].toUpperCase() && !stopwords.has(firstWord.toLowerCase()) && firstWord.length > 1) {
                                                    const yr = yrMatch ? yrMatch[1] : year
                                                    label = yr ? `${firstWord} (${yr})` : firstWord
                                                    if (page && page > 0) label += ` p.${page}`
                                                } else {
                                                    // Clean fallback: "Source X (Year)" 
                                                    label = year ? `Source ${j + 1} (${year})` : `Source ${j + 1}`
                                                    if (page && page > 0) label += ` p.${page}`
                                                }
                                            } else {
                                                label = year ? `Source ${j + 1} (${year})` : `Source ${j + 1}`
                                                if (page && page > 0) label += ` p.${page}`
                                            }
                                            const filePath = src?.source || src?.path || src?.rel_path ||
                                                (src?.metadata?.source) || (src?.metadata?.path) || (src?.metadata?.rel_path) || ''
                                            const handleOpenSource = () => {
                                                if (!filePath) return
                                                fetch(`http://localhost:${(window as any).__EDITH_BACKEND_PORT || 8003}/api/file/open`, {
                                                    method: 'POST',
                                                    headers: { 'Content-Type': 'application/json' },
                                                    body: JSON.stringify({ path: filePath })
                                                }).catch(() => { })
                                            }
                                            return (
                                                <span key={j}
                                                    className="badge badge--blue"
                                                    title={filePath || title}
                                                    style={{ cursor: filePath ? 'pointer' : 'default' }}
                                                    onClick={handleOpenSource}
                                                >
                                                    {label}
                                                </span>
                                            )
                                        })}
                                    </div>
                                )}
                            </div>
                        ))}
                        {followUps.length > 0 && !isStreaming && (
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 'var(--space-2)', marginTop: 'var(--space-2)' }}>
                                {followUps.slice(0, 3).map((fu, i) => (
                                    <button key={i} className="btn btn--sm" onClick={() => { setInput(fu); setFollowUps([]) }}
                                        style={{ maxWidth: 300, textAlign: 'left', lineHeight: 1.4 }}>
                                        {fu}
                                    </button>
                                ))}
                            </div>
                        )}
                        <div ref={bottomRef} />
                    </div>

                    {/* Smart Intent Suggestions */}
                    <SuggestionsBar input={input} onNavigate={(panel) => setActiveTab(panel as Parameters<typeof setActiveTab>[0])} />

                    {/* Input */}
                    <div className="chat-input">
                        <textarea
                            ref={inputRef}
                            value={input}
                            onChange={e => {
                                setInput(e.target.value)
                                // Auto-resize
                                const ta = e.target
                                ta.style.height = 'auto'
                                ta.style.height = Math.min(ta.scrollHeight, 144) + 'px' // max ~6 rows
                            }}
                            onKeyDown={handleKeyDown}
                            placeholder="Ask Winnie..."
                            rows={1}
                            disabled={isStreaming}
                            style={{ resize: 'none', overflow: 'hidden' }}
                        />
                        <button className="btn btn--primary" onClick={sendMessage} disabled={isStreaming || !input.trim()}>
                            Send
                        </button>
                    </div>
                </div> {/* winnie-layout__main */}
            </div> {/* winnie-layout */}
        </div>
    )
}
