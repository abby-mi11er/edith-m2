import { useState, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import { useSuggestions, SuggestionChips } from '../components/useSuggestions'
import { apiUrl } from '../api'

/* ── Paper Dive ────────────────────────────────────── */

interface DiveSection { title: string; content: string }

function parseSections(data: any): DiveSection[] {
    const parsed: DiveSection[] = []
    const payload = data?.result || data
    const labels = ['Summary', 'Methodology', 'Key Findings', 'Contributions', 'Limitations', 'Related Work']

    if (payload?.sections && typeof payload.sections === 'object') {
        if (Array.isArray(payload.sections)) {
            for (const row of payload.sections) {
                if (row?.title && row?.content) {
                    parsed.push({ title: String(row.title), content: String(row.content) })
                }
            }
        } else {
            Object.entries(payload.sections).forEach(([key, val]) => {
                parsed.push({ title: key, content: String(val || '') })
            })
        }
    } else if (payload && typeof payload === 'object') {
        labels.forEach(label => {
            const key = label.toLowerCase().replace(/\s+/g, '_')
            if (payload[key]) parsed.push({ title: label, content: String(payload[key]) })
        })
        if (parsed.length === 0 && payload.result) {
            parsed.push({ title: 'Analysis', content: String(payload.result) })
        }
    }
    return parsed
}

export default function PaperDivePanel() {
    const [paperInput, setPaperInput] = useState('')
    const [sections, setSections] = useState<DiveSection[]>([])
    const [followUp, setFollowUp] = useState('')
    const [followUpAnswer, setFollowUpAnswer] = useState('')
    const [peerReview, setPeerReview] = useState('')
    const [loading, setLoading] = useState(false)
    const diveSuggestions = useSuggestions(sections.map(s => s.title).join(', '), 3)

    const startDive = useCallback(async () => {
        const query = paperInput.trim()
        if (!query || loading) return
        setLoading(true)
        setSections([])
        setFollowUpAnswer('')
        console.log('[PaperDive] Starting deep dive:', query)

        try {
            const res = await fetch(apiUrl('/api/deep-dive/start'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: query, title: query, wait_seconds: 30 }),
            })
            if (!res.ok) {
                const errData = await res.json().catch(() => ({}))
                const detail = errData.detail || errData.error || `Server error (${res.status})`
                console.warn('[PaperDive] Backend error:', res.status, detail)
                throw new Error(detail)
            }
            const startData = await res.json()
            console.log('[PaperDive] Start response:', startData)

            // If the start response itself has sections/result (inline), parse directly
            if (startData?.result?.sections || startData?.sections) {
                setSections(parseSections(startData))
                setLoading(false)
                return
            }

            // Otherwise poll for the result using the job_id
            const jobId = startData?.job_id
            if (!jobId) throw new Error('No job_id returned')

            let attempts = 0
            const maxAttempts = 30 // ~60 seconds max
            const poll = async (): Promise<DiveSection[]> => {
                while (attempts < maxAttempts) {
                    attempts++
                    await new Promise(r => setTimeout(r, 2000))
                    const statusRes = await fetch(apiUrl(`/api/deep-dive/result?job_id=${encodeURIComponent(jobId)}`))
                    if (statusRes.ok) {
                        const resultData = await statusRes.json()
                        if (resultData?.result) {
                            return parseSections(resultData)
                        }
                        if (resultData?.status === 'failed') {
                            throw new Error('Deep dive failed')
                        }
                    } else if (statusRes.status === 404) {
                        // Job still running, keep polling
                    } else {
                        throw new Error('Failed to fetch result')
                    }
                }
                throw new Error('Deep dive timed out')
            }

            const parsed = await poll()
            setSections(parsed)
        } catch (err) {
            console.error('[PaperDive] Error:', err)
            setSections([{ title: 'Error', content: err instanceof Error ? err.message : 'Failed' }])
        }
        setLoading(false)
    }, [paperInput, loading])

    const askFollowUp = useCallback(async () => {
        if (!followUp.trim() || loading) return
        setLoading(true)
        try {
            const res = await fetch(apiUrl('/chat'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    messages: [{
                        role: 'user',
                        content: `Paper context: ${paperInput}\n\nQuestion: ${followUp.trim()}`,
                    }],
                    mode: 'grounded',
                }),
            })
            if (res.ok) {
                const data = await res.json()
                setFollowUpAnswer(data.content || data.answer || JSON.stringify(data))
            }
        } catch { /* offline */ }
        setLoading(false)
        setFollowUp('')
    }, [followUp, paperInput, loading])

    const getPeerReview = useCallback(async () => {
        if (!paperInput.trim() || loading) return
        setLoading(true)
        try {
            const draftText = sections.map(s => `## ${s.title}\n${s.content}`).join('\n\n')
            const res = await fetch(apiUrl('/api/peer-review'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ draft: `Paper: ${paperInput}\n\n${draftText}` }),
            })
            if (res.ok) {
                const data = await res.json()
                setPeerReview(data.synthesis || data.review || data.result || JSON.stringify(data, null, 2))
            }
        } catch { /* offline */ }
        setLoading(false)
    }, [paperInput, sections, loading])

    return (
        <div className="panel">
            <div style={{ padding: 'var(--space-4) var(--space-6)', borderBottom: '1px solid var(--border-primary)', display: 'flex', gap: 'var(--space-2)' }}>
                <input
                    className="input"
                    type="text"
                    placeholder="Enter a paper title or paste an abstract..."
                    value={paperInput}
                    onChange={e => setPaperInput(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && startDive()}
                />
                <button className="btn btn--primary" onClick={startDive} disabled={loading}>
                    {loading ? 'Analyzing...' : 'Deep Dive'}
                </button>
                <input type="file" id="ocr-upload" accept="image/*,.pdf" style={{ display: 'none' }}
                    onChange={async (e) => {
                        const file = e.target.files?.[0]
                        if (!file) return
                        setLoading(true)
                        try {
                            const filePath = (file as any).path ? String((file as any).path) : ''
                            const textFallback = !filePath ? await file.text().catch(() => '') : ''
                            const res = await fetch(apiUrl('/api/tools/ocr'), {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify(filePath ? { image_path: filePath } : { text: textFallback }),
                            })
                            if (res.ok) {
                                const data = await res.json()
                                const text = data.text || data.content || ''
                                setPaperInput(text.slice(0, 200))
                                // Auto-start dive with OCR text
                                const diveRes = await fetch(apiUrl('/api/deep-dive/start'), {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({ question: text.slice(0, 200), title: text.slice(0, 200), content: text, wait_seconds: 30 }),
                                })
                                if (diveRes.ok) {
                                    const diveData = await diveRes.json()
                                    const parsed = parseSections(diveData)
                                    setSections(parsed.length > 0 ? parsed : [{ title: 'Analysis', content: JSON.stringify(diveData, null, 2) }])
                                }
                            }
                        } catch { /* offline */ }
                        setLoading(false)
                        e.target.value = ''
                    }}
                />
                <button className="btn btn--sm" onClick={() => document.getElementById('ocr-upload')?.click()} disabled={loading}
                    title="Upload image or PDF for OCR">
                    Scan
                </button>
            </div>

            <div className="panel__body">
                {sections.length === 0 && !loading && (
                    <div className="placeholder">
                        <span className="placeholder__title">Paper Deep Dive</span>
                        <span className="placeholder__text">
                            Enter a paper title for a comprehensive breakdown: summary, methodology, findings, contributions, limitations, and related work.
                        </span>
                    </div>
                )}

                {loading && sections.length === 0 && (
                    <div className="placeholder">
                        <span className="placeholder__text">Analyzing paper...</span>
                    </div>
                )}

                {sections.map((s, i) => (
                    <div key={i} className="card" style={{ marginBottom: 'var(--space-3)' }}>
                        <div style={{ fontWeight: 600, marginBottom: 'var(--space-2)', color: 'var(--text-primary)' }}>
                            {s.title}
                        </div>
                        <div style={{ fontSize: 'var(--text-sm)', lineHeight: 1.7, color: 'var(--text-secondary)' }}>
                            <ReactMarkdown>{s.content}</ReactMarkdown>
                        </div>
                    </div>
                ))}

                {sections.length > 0 && !peerReview && (
                    <button className="btn btn--sm" onClick={getPeerReview} disabled={loading}
                        style={{ marginBottom: 'var(--space-4)' }}>
                        {loading ? 'Reviewing...' : 'Get Peer Review'}
                    </button>
                )}

                {peerReview && (
                    <div className="card" style={{ marginBottom: 'var(--space-4)' }}>
                        <div style={{ fontWeight: 600, marginBottom: 'var(--space-2)' }}>Peer Review</div>
                        <div style={{ fontSize: 'var(--text-sm)', lineHeight: 1.7, color: 'var(--text-secondary)' }}>
                            <ReactMarkdown>{peerReview}</ReactMarkdown>
                        </div>
                    </div>
                )}

                {sections.length > 0 && (
                    <div style={{ marginTop: 'var(--space-4)' }}>
                        <SuggestionChips
                            suggestions={diveSuggestions}
                            onSelect={text => setFollowUp(text)}
                        />
                        <div style={{ display: 'flex', gap: 'var(--space-2)', marginTop: 'var(--space-2)' }}>
                            <input
                                className="input"
                                type="text"
                                placeholder="Ask a follow-up question about this paper..."
                                value={followUp}
                                onChange={e => setFollowUp(e.target.value)}
                                onKeyDown={e => e.key === 'Enter' && askFollowUp()}
                            />
                            <button className="btn btn--primary btn--sm" onClick={askFollowUp} disabled={loading}>Ask</button>
                        </div>
                        {followUpAnswer && (
                            <div className="card" style={{ marginTop: 'var(--space-3)' }}>
                                <ReactMarkdown>{followUpAnswer}</ReactMarkdown>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    )
}
