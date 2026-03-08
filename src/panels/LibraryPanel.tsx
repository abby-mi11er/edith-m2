import { useState, useEffect, useCallback, useMemo } from 'react'
import coursesData from '../courses.json'
import { apiUrl } from '../api'
import { useStore } from '../store'

/* ── Library ───────────────────────────────────────── */

interface Course { id: string; name: string; code?: string; folder: string; category: string }
interface Paper { title: string; filename: string; path: string; course?: string; status?: string; sha256?: string }
interface Gap { topic: string; count: number; suggestion?: string }
interface Suggestion { title: string; reason?: string }

const allCourses: Course[] = [
    ...coursesData.courses,
    ...coursesData.research_projects,
]

export default function LibraryPanel() {
    const { setActiveTab } = useStore()
    const [selectedCourse, setSelectedCourse] = useState<string | null>(null)
    const [papers, setPapers] = useState<Paper[]>([])
    const [allPapers, setAllPapers] = useState<Paper[]>([])
    const [search, setSearch] = useState('')
    const [loading, setLoading] = useState(false)
    const [gaps, setGaps] = useState<Gap[]>([])
    const [suggestions, setSuggestions] = useState<Suggestion[]>([])
    const [selectedPaper, setSelectedPaper] = useState<Paper | null>(null)

    const openPaper = useCallback((paper: Paper) => {
        // Open the file via the backend
        const filePath = (paper as any).source || paper.path || paper.filename || ''
        if (filePath) {
            // Open in system viewer (Preview)
            fetch(apiUrl('/api/file/open'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: filePath }),
            }).catch(() => {
                // Fallback: open PDF in new browser tab
                window.open(apiUrl(`/api/file?path=${encodeURIComponent(filePath)}`), '_blank', 'noopener')
            })
        }
        setSelectedPaper(selectedPaper?.filename === paper.filename ? null : paper)
    }, [selectedPaper])

    const grouped = useMemo(() => {
        const map = new Map<string, Course[]>()
        allCourses.forEach(c => {
            const cat = c.category || 'Other'
            if (!map.has(cat)) map.set(cat, [])
            map.get(cat)!.push(c)
        })
        return Array.from(map.entries())
    }, [])

    // Load paper counts lazily in background — non-blocking
    useEffect(() => {
        const controller = new AbortController()
        fetch(apiUrl('/api/library/sources'), { signal: controller.signal })
            .then(r => r.ok ? r.json() : null)
            .then(data => {
                if (data) {
                    const docs = (data.sources || data.documents || data || []) as Paper[]
                    setAllPapers(docs.filter(d => {
                        const fname = d.filename || (d as any).source?.split('/').pop() || d.path?.split('/').pop() || ''
                        const ext = fname.split('.').pop()?.toLowerCase() || ''
                        return ['pdf', 'docx', 'pptx', 'md', 'txt'].includes(ext)
                    }))
                }
            })
            .catch(() => { })
        return () => controller.abort()
    }, [])

    const coursePaperCount = useCallback((courseId: string) => {
        return allPapers.filter(p => p.course === courseId).length
    }, [allPapers])

    const loadPapers = useCallback(async (courseId?: string) => {
        setLoading(true)
        try {
            const url = courseId
                ? apiUrl(`/api/library/sources?course=${encodeURIComponent(courseId)}`)
                : apiUrl('/api/library/sources')
            const res = await fetch(url)
            if (res.ok) {
                const data = await res.json()
                const docs = (data.sources || data.documents || data || []) as Paper[]
                const filteredDocs = docs.filter(d => {
                    const fname = d.filename || (d as any).source?.split('/').pop() || d.path?.split('/').pop() || ''
                    const ext = fname.split('.').pop()?.toLowerCase() || ''
                    return ['pdf', 'docx', 'pptx', 'md', 'txt'].includes(ext)
                })
                setPapers(filteredDocs)
                if (filteredDocs.length === 0) setSuggestions([])
            }
        } catch { /* offline */ }
        setLoading(false)
    }, [])

    // Load gaps once on mount
    useEffect(() => {
        fetch(apiUrl('/api/library/gaps'))
            .then(r => r.ok ? r.json() : null)
            .then(data => { if (data?.gaps) setGaps(data.gaps.slice(0, 3)) })
            .catch(() => { })
    }, [])

    // Load suggestions when a paper is selected or course changes
    useEffect(() => {
        if (papers.length > 0 && papers[0]?.sha256) {
            fetch(apiUrl(`/api/suggestions?sha256=${papers[0].sha256}&limit=3`))
                .then(r => r.ok ? r.json() : null)
                .then(data => {
                    if (data?.suggestions) setSuggestions(data.suggestions)
                    else if (data?.results) setSuggestions(data.results)
                })
                .catch(() => { })
        }
    }, [papers])

    useEffect(() => { if (selectedCourse) loadPapers(selectedCourse) }, [selectedCourse, loadPapers])

    const exportBibtex = useCallback(async () => {
        try {
            const res = await fetch(apiUrl('/api/export/bibtex'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    papers: papers.map(p => ({ title: p.title, filename: p.filename })),
                }),
            })
            if (res.ok) {
                const data = await res.json()
                const bibtex = data.bibtex || data.result || ''
                navigator.clipboard.writeText(bibtex)
            }
        } catch { /* offline */ }
    }, [papers])

    const displayPapers = selectedCourse ? papers : allPapers
    const filtered = displayPapers.filter(p =>
        !search || p.title?.toLowerCase().includes(search.toLowerCase()) ||
        p.filename?.toLowerCase().includes(search.toLowerCase())
    )

    return (
        <div className="panel">
            <div className="layout-sidebar">
                <div className="sidebar">
                    <button
                        className={`sidebar__item ${!selectedCourse ? 'sidebar__item--active' : ''}`}
                        onClick={() => setSelectedCourse(null)}
                    >
                        All Papers {allPapers.length > 0 && <span style={{ color: 'var(--text-tertiary)', fontWeight: 400 }}>({allPapers.length})</span>}
                    </button>
                    {grouped.map(([category, courses]) => (
                        <div key={category}>
                            <div className="sidebar__heading">{category}</div>
                            {courses.map(c => (
                                <button
                                    key={c.id}
                                    className={`sidebar__item ${selectedCourse === c.id ? 'sidebar__item--active' : ''}`}
                                    onClick={() => setSelectedCourse(c.id)}
                                >
                                    {c.name} {coursePaperCount(c.id) > 0 && <span style={{ color: 'var(--text-tertiary)', fontWeight: 400 }}>({coursePaperCount(c.id)})</span>}
                                </button>
                            ))}
                        </div>
                    ))}
                </div>

                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
                    <div style={{ padding: 'var(--space-3) var(--space-6)', borderBottom: '1px solid var(--border-primary)', display: 'flex', gap: 'var(--space-2)', alignItems: 'center' }}>
                        <input
                            className="input"
                            type="text"
                            placeholder="Search papers..."
                            value={search}
                            onChange={e => setSearch(e.target.value)}
                            style={{ flex: 1 }}
                        />
                        {papers.length > 0 && (
                            <button className="btn btn--sm" onClick={exportBibtex}>Export BibTeX</button>
                        )}
                    </div>

                    {/* Literature gaps banner */}
                    {gaps.length > 0 && !selectedCourse && (
                        <div style={{
                            padding: 'var(--space-3) var(--space-6)',
                            fontSize: 'var(--text-sm)',
                            color: 'var(--text-secondary)',
                            borderBottom: '1px solid var(--border-secondary)',
                            background: 'var(--bg-secondary)',
                        }}>
                            <span style={{ fontWeight: 500 }}>Gaps in your library: </span>
                            {gaps.map((g, i) => (
                                <span key={i}>
                                    {g.topic} ({g.count} paper{g.count !== 1 ? 's' : ''})
                                    {i < gaps.length - 1 ? ', ' : ''}
                                </span>
                            ))}
                        </div>
                    )}

                    <div className="panel__body">
                        {loading && <div className="placeholder"><span className="placeholder__text">Loading...</span></div>}
                        {!loading && filtered.length === 0 && (
                            <div className="placeholder">
                                <span className="placeholder__title">No papers indexed yet</span>
                                <span className="placeholder__text">
                                    Run the indexer to scan your course folders and populate the library.
                                </span>
                            </div>
                        )}
                        {filtered.map((p, i) => (
                            <div key={i} className="card card--interactive" style={{ marginBottom: 'var(--space-2)' }}
                                onClick={() => openPaper(p)}>
                                <div style={{ fontWeight: 500 }}>{p.title || p.filename}</div>
                                {p.course && (
                                    <span className="badge" style={{ marginTop: 4 }}>
                                        {allCourses.find(c => c.id === p.course || c.folder.toLowerCase().replace(/[^a-z0-9]/g, '_') === p.course)?.name || p.course.replace(/_/g, ' ')}
                                    </span>
                                )}
                                {selectedPaper && (selectedPaper.filename === p.filename || (selectedPaper as any).source === (p as any).source) && (
                                    <div style={{ marginTop: 'var(--space-3)', paddingTop: 'var(--space-3)', borderTop: '1px solid var(--border-secondary)', display: 'flex', gap: 'var(--space-2)', flexWrap: 'wrap', alignItems: 'center' }}>
                                        <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-tertiary)', flex: 1 }}>
                                            {(p as any).source || p.path || p.filename}
                                        </span>
                                        <button className="btn btn--sm btn--primary" onClick={(e) => {
                                            e.stopPropagation()
                                            setActiveTab('dive' as any)
                                        }}>Paper Dive</button>
                                        <button className="btn btn--sm" onClick={(e) => {
                                            e.stopPropagation()
                                            setActiveTab('citations' as any)
                                        }}>Citations</button>
                                        <button className="btn btn--sm" onClick={(e) => {
                                            e.stopPropagation()
                                            setActiveTab('chat' as any)
                                        }}>Ask Winnie</button>
                                    </div>
                                )}
                            </div>
                        ))}

                        {/* Smart reading suggestions */}
                        {suggestions.length > 0 && (
                            <div style={{ marginTop: 'var(--space-6)' }}>
                                <div style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 'var(--space-2)' }}>
                                    Suggested Reading
                                </div>
                                {suggestions.map((s, i) => (
                                    <div key={i} className="card" style={{ marginBottom: 'var(--space-2)' }}>
                                        <div style={{ fontWeight: 500, fontSize: 'var(--text-sm)' }}>{s.title}</div>
                                        {s.reason && (
                                            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-tertiary)', marginTop: 2 }}>
                                                {s.reason}
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}
