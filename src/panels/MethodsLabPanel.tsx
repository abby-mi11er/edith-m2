import { useState, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import { apiUrl } from '../api'

/* ── Methods Lab ───────────────────────────────────── */

type LabMode = 'learn' | 'analyze'

interface Method {
    id: string
    name: string
    description: string
}

const METHODS: Method[] = [
    { id: 'did', name: 'Difference-in-Differences', description: 'Parallel trends and causal effects' },
    { id: 'iv', name: 'Instrumental Variables', description: 'Endogeneity via exclusion restriction' },
    { id: 'rdd', name: 'Regression Discontinuity', description: 'Assignment thresholds' },
    { id: 'synth', name: 'Synthetic Control', description: 'Counterfactual from donor pool' },
    { id: 'fe', name: 'Fixed Effects', description: 'Time-invariant unobservables' },
    { id: 'psm', name: 'Propensity Score Matching', description: 'Covariate balance' },
    { id: 'event', name: 'Event Study', description: 'Dynamic treatment effects' },
    { id: 'ols', name: 'OLS Regression', description: 'The workhorse' },
]

export default function MethodsLabPanel() {
    const [mode, setMode] = useState<LabMode>('learn')
    const [selectedMethod, setSelectedMethod] = useState<string | null>(null)
    const [question, setQuestion] = useState('')
    const [answer, setAnswer] = useState('')
    const [analyzeInput, setAnalyzeInput] = useState('')
    const [auditResult, setAuditResult] = useState('')
    const [loading, setLoading] = useState(false)
    const [socratic, setSocratic] = useState(false)
    const [socraticSessionId, setSocraticSessionId] = useState<string | null>(null)

    const askMethod = useCallback(async () => {
        if (!question.trim() || !selectedMethod || loading) return
        setLoading(true)
        setAnswer('')
        try {
            if (socratic) {
                const endpoint = socraticSessionId ? '/api/socratic/respond' : '/api/socratic/start'
                const methodName = METHODS.find(m => m.id === selectedMethod)?.name || selectedMethod
                const body = socraticSessionId
                    ? { session_id: socraticSessionId, response: question.trim() }
                    : { topic: `${methodName}: ${question.trim()}`, question: question.trim() }
                const res = await fetch(apiUrl(endpoint), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body),
                })
                if (res.ok) {
                    const data = await res.json()
                    if (data.session_id) setSocraticSessionId(data.session_id)
                    setAnswer(data.question || data.guidance || data.response || JSON.stringify(data))
                }
            } else {
                const res = await fetch(apiUrl('/api/method/decode'), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ method: selectedMethod, text: question.trim() }),
                })
                if (res.ok) {
                    const data = await res.json()
                    const payload = data.analysis || data.explanation || data.answer || data
                    if (typeof payload === 'string') {
                        setAnswer(payload)
                    } else if (payload?.raw_analysis) {
                        // LLM returned structured JSON with a readable analysis field
                        setAnswer(payload.raw_analysis)
                    } else if (typeof payload === 'object' && payload !== null) {
                        // Convert structured JSON keys to readable markdown
                        const parts: string[] = []
                        for (const [key, val] of Object.entries(payload)) {
                            if (val && typeof val === 'string') {
                                const label = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
                                parts.push(`**${label}:** ${val}`)
                            } else if (Array.isArray(val) && val.length > 0) {
                                const label = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
                                parts.push(`**${label}:**\n${val.map(v => `- ${String(v)}`).join('\n')}`)
                            }
                        }
                        setAnswer(parts.length > 0 ? parts.join('\n\n') : JSON.stringify(payload, null, 2))
                    } else {
                        setAnswer(String(payload))
                    }
                }
            }
        } catch { /* offline */ }
        setLoading(false)
    }, [question, selectedMethod, loading, socratic, socraticSessionId])

    const analyzePaper = useCallback(async () => {
        if (!analyzeInput.trim() || loading) return
        setLoading(true)
        setAuditResult('')
        try {
            const res = await fetch(apiUrl('/api/sniper/audit'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ paper_text: analyzeInput.trim() }),
            })
            if (res.ok) {
                const data = await res.json()
                setAuditResult(data.report || data.audit || JSON.stringify(data, null, 2))
            }
        } catch { /* offline */ }
        setLoading(false)
    }, [analyzeInput, loading])

    return (
        <div className="panel">
            <div className="mode-bar">
                <button className={`btn btn--sm btn--pill ${mode === 'learn' ? 'active' : ''}`}
                    onClick={() => setMode('learn')}>Learn a Method</button>
                <button className={`btn btn--sm btn--pill ${mode === 'analyze' ? 'active' : ''}`}
                    onClick={() => setMode('analyze')}>Analyze a Paper</button>
                {mode === 'learn' && (
                    <label style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6, fontSize: 'var(--text-sm)', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                        <input type="checkbox" checked={socratic} onChange={e => { setSocratic(e.target.checked); setSocraticSessionId(null) }} />
                        Socratic
                    </label>
                )}
            </div>

            <div className="panel__body">
                {mode === 'learn' ? (
                    <>
                        {!selectedMethod ? (
                            <div className="grid-cards">
                                {METHODS.map(m => (
                                    <div key={m.id} className="card card--interactive" onClick={() => setSelectedMethod(m.id)}>
                                        <div style={{ fontWeight: 600, marginBottom: 4 }}>{m.name}</div>
                                        <div style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>{m.description}</div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', marginBottom: 'var(--space-5)' }}>
                                    <button className="btn btn--sm btn--ghost" onClick={() => { setSelectedMethod(null); setAnswer(''); setSocraticSessionId(null) }}>
                                        Back
                                    </button>
                                    <span style={{ fontWeight: 600 }}>{METHODS.find(m => m.id === selectedMethod)?.name}</span>
                                </div>

                                <div style={{ display: 'flex', gap: 'var(--space-2)', marginBottom: 'var(--space-4)' }}>
                                    <input
                                        className="input"
                                        type="text"
                                        placeholder={socratic ? "Answer the question or ask for a hint..." : "Ask a question about this method..."}
                                        value={question}
                                        onChange={e => setQuestion(e.target.value)}
                                        onKeyDown={e => e.key === 'Enter' && askMethod()}
                                    />
                                    <button className="btn btn--primary" onClick={askMethod} disabled={loading}>
                                        {loading ? 'Thinking...' : 'Ask'}
                                    </button>
                                </div>

                                {answer && (
                                    <div className="card">
                                        <ReactMarkdown>{answer}</ReactMarkdown>
                                        <div style={{ display: 'flex', gap: 'var(--space-2)', marginTop: 'var(--space-3)', borderTop: '1px solid var(--border)', paddingTop: 'var(--space-3)' }}>
                                            <button className="btn btn--sm" disabled={loading} onClick={async () => {
                                                try {
                                                    const res = await fetch(apiUrl('/api/tools/flashcard'), {
                                                        method: 'POST',
                                                        headers: { 'Content-Type': 'application/json' },
                                                        body: JSON.stringify({
                                                            question: question || `Explain ${METHODS.find(m => m.id === selectedMethod)?.name || 'this method'}`,
                                                            answer,
                                                            source: selectedMethod,
                                                        }),
                                                    })
                                                    if (res.ok) {
                                                        const data = await res.json()
                                                        const cards = data.cards || data.flashcards || []
                                                        setAnswer(prev => prev + '\n\n---\n\n**Flashcards:**\n' + cards.map((c: { q: string, a: string }, i: number) => `${i + 1}. **Q:** ${c.q}\n   **A:** ${c.a}`).join('\n\n'))
                                                    }
                                                } catch { /* offline */ }
                                            }}>Make Flashcards</button>
                                            <button className="btn btn--sm" disabled={loading} onClick={async () => {
                                                try {
                                                    const res = await fetch(apiUrl('/api/streams/bridge/method-to-code'), {
                                                        method: 'POST',
                                                        headers: { 'Content-Type': 'application/json' },
                                                        body: JSON.stringify({
                                                            method: {
                                                                id: selectedMethod,
                                                                name: METHODS.find(m => m.id === selectedMethod)?.name || selectedMethod,
                                                                explanation: answer,
                                                                language: 'r',
                                                            },
                                                        }),
                                                    })
                                                    if (res.ok) {
                                                        const data = await res.json()
                                                        setAnswer(prev => prev + '\n\n---\n\n**R Code:**\n```r\n' + (data.code || data.r_code || '') + '\n```')
                                                    }
                                                } catch { /* offline */ }
                                            }}>Translate to R</button>
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}
                    </>
                ) : (
                    <div>
                        <div style={{ display: 'flex', gap: 'var(--space-2)', marginBottom: 'var(--space-4)' }}>
                            <input
                                className="input"
                                type="text"
                                placeholder="Enter a paper title or DOI for forensic analysis..."
                                value={analyzeInput}
                                onChange={e => setAnalyzeInput(e.target.value)}
                                onKeyDown={e => e.key === 'Enter' && analyzePaper()}
                            />
                            <button className="btn btn--primary" onClick={analyzePaper} disabled={loading}>
                                {loading ? 'Analyzing...' : 'Analyze'}
                            </button>
                        </div>

                        {auditResult && (
                            <div className="card">
                                <ReactMarkdown>{auditResult}</ReactMarkdown>
                            </div>
                        )}

                        {!auditResult && !loading && (
                            <div className="placeholder">
                                <span className="placeholder__title">Forensic Method Analysis</span>
                                <span className="placeholder__text">
                                    Enter a paper title to identify the method, evaluate the identification strategy, and check robustness assumptions.
                                </span>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    )
}
