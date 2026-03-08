import { useState, useCallback } from 'react'
import { apiUrl, backendUrl } from '../api'

/* ── Search ────────────────────────────────────────── */

interface SearchResult {
    id?: string
    openalexId?: string
    title: string
    authors?: string
    year?: string
    abstract?: string
    doi?: string
    url?: string
    source: string
}

type SearchSource = 'all' | 'openalex' | 'local' | 'scholar' | 'nyt'

const SOURCES: { id: SearchSource; label: string }[] = [
    { id: 'all', label: 'All Sources' },
    { id: 'openalex', label: 'Academic' },
    { id: 'local', label: 'My Library' },
    { id: 'scholar', label: 'Scholar' },
    { id: 'nyt', label: 'News' },
]

function parseOpenAlexId(value: string): string {
    const trimmed = String(value || '').trim()
    if (!trimmed) return ''
    const match = trimmed.match(/\/([A-Z]\d+)$/i)
    return match ? match[1].toUpperCase() : trimmed
}

function toSearchResult(raw: any, sourceHint = ''): SearchResult | null {
    if (!raw || typeof raw !== 'object') return null

    const title = String(raw.title || raw.display_name || raw.headline || raw.name || '').trim()
    if (!title) return null

    const authorList = Array.isArray(raw.authors)
        ? raw.authors
        : Array.isArray(raw.authorships)
            ? raw.authorships
            : []

    const authors = Array.isArray(authorList)
        ? authorList
            .map((a: any) => {
                if (typeof a === 'string') return a
                return a?.name || a?.display_name || a?.author?.display_name || ''
            })
            .filter(Boolean)
            .slice(0, 5)
            .join(', ')
        : String(raw.authors || raw.author || '').trim()

    const source = String(
        sourceHint ||
        raw.source ||
        raw.source_name ||
        raw.host_venue?.display_name ||
        raw.journal ||
        'search',
    ).trim()

    const openalexId = parseOpenAlexId(String(raw.id || raw.openalex_id || raw.openalexId || ''))

    return {
        id: String(raw.id || raw.work_id || openalexId || ''),
        openalexId: openalexId || undefined,
        title,
        authors,
        year: String(raw.year || raw.publication_year || raw.pub_year || '').trim() || undefined,
        abstract: String(raw.abstract || raw.abstract_text || raw.summary || '').trim() || undefined,
        doi: String(raw.doi || '').replace(/^https?:\/\/doi\.org\//i, '').trim() || undefined,
        url: String(raw.url || raw.primary_location?.landing_page_url || raw.landing_page_url || '').trim() || undefined,
        source,
    }
}

function unpackResults(payload: any, sourceHint = ''): SearchResult[] {
    const out: SearchResult[] = []

    const appendMany = (items: any, itemSource = sourceHint) => {
        if (!Array.isArray(items)) return
        for (const item of items) {
            const normalized = toSearchResult(item, itemSource)
            if (normalized) out.push(normalized)
        }
    }

    if (Array.isArray(payload)) {
        appendMany(payload)
    } else if (payload && typeof payload === 'object') {
        if (Array.isArray(payload.results)) appendMany(payload.results)
        if (Array.isArray(payload.works)) appendMany(payload.works)
        if (Array.isArray(payload.articles)) appendMany(payload.articles)
        if (Array.isArray(payload.documents)) appendMany(payload.documents)

        if (payload.results && typeof payload.results === 'object' && !Array.isArray(payload.results)) {
            for (const [key, value] of Object.entries(payload.results)) {
                appendMany(value, key)
            }
        }
    }

    const dedup = new Map<string, SearchResult>()
    for (const row of out) {
        const key = `${row.openalexId || ''}|${(row.doi || '').toLowerCase()}|${row.title.toLowerCase()}`
        if (!dedup.has(key)) dedup.set(key, row)
    }

    return Array.from(dedup.values())
}

export default function SearchPanel() {
    const [source, setSource] = useState<SearchSource>('all')
    const [query, setQuery] = useState('')
    const [results, setResults] = useState<SearchResult[]>([])
    const [loading, setLoading] = useState(false)

    const search = useCallback(async () => {
        if (!query.trim()) return
        setLoading(true)
        setResults([])
        const q = encodeURIComponent(query.trim())

        const primaryBySource: Record<SearchSource, string> = {
            all: `/api/research/search?q=${q}&source=all&per_page=20`,
            openalex: `/api/openalex/search?q=${q}&per_page=20`,
            local: `/api/search?q=${q}&source=local&per_page=20`,
            scholar: `/api/scholar/search?q=${q}&limit=20`,
            nyt: `/api/nyt/search?q=${q}`,
        }

        try {
            let items: SearchResult[] = []

            const res = await fetch(apiUrl(primaryBySource[source]))
            if (res.ok) {
                const data = await res.json()
                items = unpackResults(data, source === 'all' ? '' : source)
            }

            if (items.length === 0 && source !== 'openalex') {
                const fallbacks = [
                    `/api/openalex/search?q=${q}&per_page=20`,
                    `/api/scholar/search?q=${q}&limit=20`,
                    `/api/crossref/search?q=${q}&limit=20`,
                ]
                for (const url of fallbacks) {
                    try {
                        const fb = await fetch(apiUrl(url))
                        if (!fb.ok) continue
                        const data = await fb.json()
                        items = unpackResults(data)
                        if (items.length > 0) break
                    } catch {
                        // try next fallback
                    }
                }
            }

            setResults(items)
        } catch {
            // offline
        }

        setLoading(false)
    }, [query, source])

    const addToLibrary = useCallback(async (result: SearchResult) => {
        if (!result.openalexId) {
            if (result.url) window.open(result.url, '_blank', 'noopener')
            return
        }
        try {
            await fetch(apiUrl('/api/openalex/import'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ openalex_id: result.openalexId }),
            })
        } catch {
            // offline
        }
    }, [])

    return (
        <div className="panel">
            <div className="mode-bar">
                {SOURCES.map(s => (
                    <button
                        key={s.id}
                        className={`btn btn--sm btn--pill ${source === s.id ? 'active' : ''}`}
                        onClick={() => { setSource(s.id); setResults([]) }}
                    >
                        {s.label}
                    </button>
                ))}
                <div style={{ flex: 1 }} />
                <button className="btn btn--sm btn--ghost" onClick={async () => {
                    try {
                        const statusRes = await fetch(apiUrl('/api/mendeley/status'))
                        if (statusRes.ok) {
                            const status = await statusRes.json()
                            const isConnected = Boolean(status.connected || status.authenticated)
                            if (!isConnected) {
                                window.open(backendUrl('/oauth/mendeley/start'), '_blank', 'noopener')
                                return
                            }
                        }
                        await fetch(apiUrl('/api/mendeley/sync'), { method: 'POST' })
                    } catch {
                        // offline
                    }
                }}>Sync Mendeley</button>
            </div>

            <div style={{ padding: 'var(--space-4) var(--space-6)', display: 'flex', gap: 'var(--space-2)' }}>
                <input
                    className="input"
                    type="text"
                    placeholder="Search papers, Semantic Scholar, news..."
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && search()}
                />
                <button className="btn btn--primary" onClick={search} disabled={loading}>
                    {loading ? 'Searching...' : 'Search'}
                </button>
            </div>

            <div className="panel__body">
                {results.length === 0 && !loading && (
                    <div className="placeholder">
                        <span className="placeholder__title">Search across all sources at once</span>
                        <span className="placeholder__text">
                            OpenAlex, Semantic Scholar, your library, and NYT
                        </span>
                    </div>
                )}
                {results.map((r, i) => (
                    <div key={`${r.openalexId || r.doi || r.title}-${i}`} className="card" style={{ marginBottom: 'var(--space-3)' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                            <div style={{ flex: 1 }}>
                                <div style={{ fontWeight: 600, fontSize: 'var(--text-base)' }}>{r.title}</div>
                                <div style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)', marginTop: 2 }}>
                                    {r.authors}{r.year && ` (${r.year})`}
                                    {r.source && <span className="badge" style={{ marginLeft: 8 }}>{r.source}</span>}
                                </div>
                                {r.abstract && (
                                    <div style={{ fontSize: 'var(--text-sm)', color: 'var(--text-tertiary)', marginTop: 'var(--space-2)', lineHeight: 1.6 }}>
                                        {r.abstract.slice(0, 200)}...
                                    </div>
                                )}
                            </div>
                            <button className="btn btn--sm" onClick={() => addToLibrary(r)}>
                                {r.openalexId ? 'Import' : 'Open'}
                            </button>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    )
}
