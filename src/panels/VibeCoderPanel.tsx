import { useState, useRef, useEffect, useCallback } from 'react'
import { useSuggestions, SuggestionChips } from '../components/useSuggestions'
import { apiUrl } from '../api'
import { useStore } from '../store'

/* ── Vibe Coder ────────────────────────────────────── */


interface Dataset { name: string; path: string; size?: string }
const DATASET_EXCLUDE = ['node_modules', '/dist/', '/release/', '/.git/', '/electron/']

export default function VibeCoderPanel() {
    const { vibeMessages: messages, vibeLang: lang, addVibeMessage, setVibeLang } = useStore()
    const setLang = setVibeLang
    const [input, setInput] = useState('')
    const [loading, setLoading] = useState(false)
    const [datasets, setDatasets] = useState<Dataset[]>([])
    const bottomRef = useRef<HTMLDivElement>(null)
    const latestAssistantMessage = messages.filter(m => m.role === 'assistant').at(-1)?.content || ''
    const suggestions = useSuggestions(latestAssistantMessage, 3)

    useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

    // Discover available datasets on mount
    useEffect(() => {
        fetch(apiUrl('/api/vibe/datasets'))
            .then(r => r.ok ? r.json() : null)
            .then(data => {
                if (!data?.datasets) return
                const cleaned = (data.datasets as Dataset[]).filter(d =>
                    !DATASET_EXCLUDE.some(ex => d.path?.includes(ex)),
                )
                setDatasets(cleaned)
            })
            .catch(() => { })
    }, [])

    const generate = useCallback(async () => {
        const text = input.trim()
        if (!text || loading) return
        setInput('')
        addVibeMessage({ role: 'user', content: text })
        setLoading(true)

        try {
            const res = await fetch(apiUrl('/api/vibe/generate'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    directive: text,
                    language: lang,
                    history: messages.map(m => ({ role: m.role, content: m.content })),
                }),
            })
            if (!res.ok) throw new Error('Generation failed')
            const data = await res.json()
            addVibeMessage({
                role: 'assistant',
                content: data.generated?.code || data.code || data.result || '',
                language: lang,
            })
        } catch (err) {
            addVibeMessage({
                role: 'assistant',
                content: `Error: ${err instanceof Error ? err.message : 'Failed'}`,
            })
        }
        setLoading(false)
    }, [input, lang, messages, loading])

    const copyCode = (code: string) => navigator.clipboard.writeText(code)

    const saveCode = (code: string) => {
        const ext = lang === 'stata' ? '.do' : lang === 'r' ? '.R' : '.py'
        const blob = new Blob([code], { type: 'text/plain' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `analysis${ext}`
        a.click()
        URL.revokeObjectURL(url)
    }

    const explainCode = useCallback(async (code: string) => {
        setLoading(true)
        try {
            const res = await fetch(apiUrl('/api/vibe/explain'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ code, language: lang }),
            })
            if (res.ok) {
                const data = await res.json()
                addVibeMessage({
                    role: 'assistant',
                    content: data.explanation || data.result || 'No explanation available.',
                })
            }
        } catch { /* offline */ }
        setLoading(false)
    }, [lang])

    const exportLatex = useCallback(async (code: string) => {
        try {
            const res = await fetch(apiUrl('/api/stata/to-latex'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ log_text: code, title: 'Regression Results' }),
            })
            if (res.ok) {
                const data = await res.json()
                const latex = data.latex || data.result || ''
                navigator.clipboard.writeText(latex)
                addVibeMessage({
                    role: 'assistant',
                    content: `LaTeX copied to clipboard:\n\n${latex}`,
                })
            }
        } catch { /* offline */ }
    }, [lang])

    return (
        <div className="panel">
            <div className="mode-bar">
                <button className={`btn btn--sm btn--pill ${lang === 'stata' ? 'active' : ''}`}
                    onClick={() => setLang('stata')}>Stata</button>
                <button className={`btn btn--sm btn--pill ${lang === 'r' ? 'active' : ''}`}
                    onClick={() => setLang('r')}>R</button>
                <button className={`btn btn--sm btn--pill ${lang === 'python' ? 'active' : ''}`}
                    onClick={() => setLang('python')}>Python</button>
                {datasets.length > 0 && (
                    <span style={{ marginLeft: 'auto', fontSize: 'var(--text-xs)', color: 'var(--text-tertiary)' }}>
                        {datasets.length} dataset{datasets.length !== 1 ? 's' : ''} available
                    </span>
                )}
            </div>

            <div className="panel__body">
                {messages.length === 0 && (
                    <div className="placeholder">
                        <span className="placeholder__title">Describe your analysis</span>
                        <span className="placeholder__text">
                            Tell me what you want to do and I'll write the {lang === 'stata' ? 'Stata' : lang === 'r' ? 'R' : 'Python'} code.
                            {datasets.length > 0 && ` I can see ${datasets.length} dataset${datasets.length !== 1 ? 's' : ''} on your drive.`}
                        </span>
                    </div>
                )}
                {messages.map((msg, i) => (
                    <div key={i} style={{ marginBottom: 'var(--space-4)' }}>
                        {msg.role === 'user' ? (
                            <div className="message message--user">
                                <div className="message__bubble">{msg.content}</div>
                            </div>
                        ) : (
                            <div>
                                <div className="code-block">{msg.content}</div>
                                {msg.language && (
                                    <div style={{ display: 'flex', gap: 'var(--space-2)', marginTop: 'var(--space-2)' }}>
                                        <button className="btn btn--sm" onClick={() => copyCode(msg.content)}>Copy</button>
                                        <button className="btn btn--sm" onClick={() => saveCode(msg.content)}>
                                            Save as {lang === 'stata' ? '.do' : lang === 'r' ? '.R' : '.py'}
                                        </button>
                                        <button className="btn btn--sm" onClick={() => explainCode(msg.content)}>Explain</button>
                                        {lang === 'stata' && (
                                            <button className="btn btn--sm" onClick={() => exportLatex(msg.content)}>
                                                Export LaTeX
                                            </button>
                                        )}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                ))}
                {loading && <div style={{ color: 'var(--text-tertiary)', fontSize: 'var(--text-sm)' }}>Generating...</div>}
                <div ref={bottomRef} />
            </div>

            <SuggestionChips
                suggestions={suggestions}
                onSelect={text => setInput(text)}
            />

            <div className="chat-input">
                <textarea
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); generate() } }}
                    placeholder={`Describe what you want to do in ${lang === 'stata' ? 'Stata' : lang === 'r' ? 'R' : 'Python'}...`}
                    rows={1}
                    disabled={loading}
                />
                <button className="btn btn--primary" onClick={generate} disabled={loading || !input.trim()}>
                    Generate
                </button>
            </div>
        </div>
    )
}
