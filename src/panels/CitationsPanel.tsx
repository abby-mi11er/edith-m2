import { useState, useCallback, useRef } from 'react'
import { apiUrl } from '../api'
import { useSuggestions, SuggestionChips } from '../components/useSuggestions'

/* ── Citations ─────────────────────────────────────── */

type CitationView = 'cited' | 'citedby' | 'connected' | 'suggestions'

interface Citation {
    title: string
    authors?: string
    year?: string
    doi?: string
    url?: string
    id?: string
}

export default function CitationsPanel() {
    const [view, setView] = useState<CitationView>('cited')
    const [query, setQuery] = useState('')
    const [citations, setCitations] = useState<Citation[]>([])
    const [loading, setLoading] = useState(false)
    const suggestionContext = `${query} ${citations.slice(0, 3).map(c => c.title).join(', ')}`.trim()
    const suggestions = useSuggestions(suggestionContext, 3)

    const exportBibtex = useCallback(async () => {
        if (citations.length === 0) return
        try {
            const res = await fetch(apiUrl('/api/export/bibtex'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    docs: citations.map(c => ({
                        title: c.title,
                        author: c.authors || '',
                        year: c.year || '',
                        doi: c.doi || '',
                    })),
                }),
            })
            if (res.ok) {
                const data = await res.json()
                navigator.clipboard.writeText(data.bibtex || data.result || '')
            }
        } catch { /* offline */ }
    }, [citations])

    const resolveSeedWork = useCallback(async (term: string) => {
        const q = term.trim()
        if (!q) return null
        try {
            const res = await fetch(apiUrl(`/api/openalex/search?q=${encodeURIComponent(q)}&per_page=1`))
            if (!res.ok) return null
            const data = await res.json()
            const first = (data.results || data.works || [])[0]
            return first || null
        } catch {
            return null
        }
    }, [])

    const toCitation = useCallback((raw: any): Citation => {
        const title = raw?.title || raw?.display_name || raw?.name || raw?.id || 'Untitled'
        const authorsRaw = raw?.authors || raw?.author || raw?.authorships || []
        let authors = ''
        if (typeof authorsRaw === 'string') {
            authors = authorsRaw
        } else if (Array.isArray(authorsRaw)) {
            const names = authorsRaw
                .map((a: any) => a?.name || a?.display_name || a?.author?.display_name || '')
                .filter(Boolean)
            authors = names.join(', ')
        }
        const yearValue = raw?.year || raw?.publication_year
        const doiValue = raw?.doi || (typeof raw?.ids?.doi === 'string' ? raw.ids.doi : '')
        const openalexId = typeof raw?.openalex_id === 'string'
            ? raw.openalex_id
            : (typeof raw?.id === 'string' && raw.id.includes('openalex.org') ? raw.id : '')
        const urlValue = raw?.url || openalexId || doiValue || ''
        return {
            title,
            authors,
            year: yearValue ? String(yearValue) : '',
            doi: doiValue ? String(doiValue).replace('https://doi.org/', '') : '',
            url: urlValue,
            id: openalexId || raw?.id || '',
        }
    }, [])

    const pendingQueryRef = useRef<string | null>(null)

    const loadCitations = useCallback(async (overrideQuery?: string) => {
        const q = (overrideQuery ?? pendingQueryRef.current ?? query).trim()
        pendingQueryRef.current = null
        if (!q || loading) return
        setLoading(true)
        setCitations([])
        try {
            const seed = await resolveSeedWork(q)
            const seedIdRaw = seed?.openalex_id || seed?.id || ''
            const seedId = String(seedIdRaw).replace('https://openalex.org/', '')

            if (view === 'suggestions') {
                const local = await fetch(apiUrl(`/api/suggestions?title=${encodeURIComponent(q)}&limit=25`))
                if (local.ok) {
                    const localData = await local.json()
                    const suggestions = (localData.suggestions || []).map(toCitation).filter((c: Citation) => c.title)
                    if (suggestions.length > 0) {
                        setCitations(suggestions)
                        setLoading(false)
                        return
                    }
                }

                const res = await fetch(apiUrl('/api/openalex/recommend'), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: q, per_page: 20 }),
                })
                if (res.ok) {
                    const data = await res.json()
                    setCitations((data.results || []).map(toCitation))
                }
                setLoading(false)
                return
            }

            if (view === 'connected') {
                const seedForGraph = seed?.doi || seedId || q
                const res = await fetch(apiUrl('/api/connectors/similarity/graph'), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ seed_paper_id: seedForGraph, depth: 1 }),
                })
                if (res.ok) {
                    const data = await res.json()
                    const nodes = Array.isArray(data.nodes) ? data.nodes : []
                    const mapped = nodes
                        .filter((n: any) => n?.id !== seedForGraph)
                        .map((n: any) => ({
                            title: n?.title || n?.id || 'Untitled',
                            authors: Array.isArray(n?.authors)
                                ? n.authors.map((a: any) => a?.name || '').filter(Boolean).join(', ')
                                : '',
                            year: n?.year ? String(n.year) : '',
                            id: n?.id || '',
                        }))
                    setCitations(mapped)
                }
                setLoading(false)
                return
            }

            if (seedId) {
                const res = await fetch(apiUrl(`/api/openalex/citations/${encodeURIComponent(seedId)}?per_page=40`))
                if (res.ok) {
                    const data = await res.json()
                    if (view === 'citedby') {
                        setCitations((data.citing || []).map(toCitation))
                    } else {
                        const refs = Array.isArray(data.referenced_ids) ? data.referenced_ids : []
                        if (refs.length > 0) {
                            // Batch-resolve referenced IDs into full paper metadata
                            try {
                                const resolveRes = await fetch(apiUrl('/api/openalex/resolve'), {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({ ids: refs.slice(0, 40) }),
                                })
                                if (resolveRes.ok) {
                                    const resolveData = await resolveRes.json()
                                    const resolved = (resolveData.results || []).map(toCitation).filter((c: Citation) => c.title)
                                    if (resolved.length > 0) {
                                        setCitations(resolved)
                                    } else {
                                        // Fallback: show short IDs
                                        setCitations(refs.map((id: string) => ({
                                            title: String(id).replace('https://openalex.org/', ''),
                                            authors: 'OpenAlex reference',
                                            year: '',
                                            doi: '',
                                            url: String(id).startsWith('http') ? String(id) : `https://openalex.org/${id}`,
                                            id: String(id),
                                        })))
                                    }
                                } else {
                                    throw new Error('Resolution failed')
                                }
                            } catch {
                                // Fallback to short IDs if batch resolution fails
                                setCitations(refs.map((id: string) => ({
                                    title: String(id).replace('https://openalex.org/', ''),
                                    authors: 'OpenAlex reference',
                                    year: '',
                                    doi: '',
                                    url: String(id).startsWith('http') ? String(id) : `https://openalex.org/${id}`,
                                    id: String(id),
                                })))
                            }
                        } else {
                            setCitations([])
                        }
                    }
                } else {
                    setCitations([])
                }
            } else {
                setCitations([])
            }
        } catch { /* offline */ }
        setLoading(false)
    }, [query, view, loading, resolveSeedWork, toCitation])

    const VIEWS: { id: CitationView; label: string }[] = [
        { id: 'cited', label: 'Works Cited' },
        { id: 'citedby', label: 'Cited By' },
        { id: 'connected', label: 'Connected Papers' },
        { id: 'suggestions', label: 'Suggestions' },
    ]

    return (
        <div className="panel">
            <div className="mode-bar">
                {VIEWS.map(v => (
                    <button
                        key={v.id}
                        className={`btn btn--sm btn--pill ${view === v.id ? 'active' : ''}`}
                        onClick={() => { setView(v.id); setCitations([]) }}
                    >
                        {v.label}
                    </button>
                ))}
            </div>

            <div style={{ padding: 'var(--space-4) var(--space-6)', display: 'flex', gap: 'var(--space-2)', borderBottom: '1px solid var(--border-secondary)' }}>
                <input
                    className="input"
                    type="text"
                    placeholder="Enter a paper title or DOI..."
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && loadCitations()}
                />
                <button className="btn btn--primary" onClick={() => loadCitations()} disabled={loading}>
                    {loading ? 'Loading...' : 'Search'}
                </button>
                {citations.length > 0 && (
                    <button className="btn btn--sm" onClick={exportBibtex}>Export BibTeX</button>
                )}
            </div>

            <div className="panel__body">
                {citations.length === 0 && !loading && (
                    <div className="placeholder">
                        <span className="placeholder__title">Citation Network</span>
                        <span className="placeholder__text">
                            Enter a paper to explore its references, citations, connected papers, and personalized suggestions.
                        </span>
                    </div>
                )}
                {citations.map((c, i) => (
                    <div key={i} className="card card--interactive" style={{ marginBottom: 'var(--space-2)', cursor: 'pointer' }}
                        onClick={() => {
                            const title = c.title
                            setQuery(title)
                            pendingQueryRef.current = title
                            // Auto-trigger search in next tick so state is updated
                            setTimeout(() => loadCitations(title), 0)
                        }}>
                        <div style={{ fontWeight: 500 }}>{c.title}</div>
                        <div style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)', marginTop: 2 }}>
                            {c.authors}{c.year && ` (${c.year})`}
                        </div>
                        {c.url && (
                            <a
                                href={c.url}
                                target="_blank"
                                rel="noreferrer"
                                style={{ display: 'inline-block', marginTop: 6, fontSize: 'var(--text-xs)' }}
                                onClick={e => e.stopPropagation()}
                            >
                                Open source
                            </a>
                        )}
                        {c.doi && (
                            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-tertiary)', marginTop: 4 }}>
                                {c.doi}
                            </div>
                        )}
                    </div>
                ))}
                {citations.length > 0 && (
                    <SuggestionChips
                        suggestions={suggestions}
                        onSelect={text => setQuery(text)}
                    />
                )}
            </div>
        </div>
    )
}
