import { useState, useEffect, useCallback, useRef } from 'react'
import { apiUrl } from '../api'

/* ── Suggestions Bar ───────────────────────────────── */

interface Suggestion {
    label: string
    action: string
    panel?: string
}

export default function SuggestionsBar({
    input,
    onNavigate,
}: {
    input: string
    onNavigate: (panel: string, prefill?: string) => void
}) {
    const [suggestions, setSuggestions] = useState<Suggestion[]>([])
    const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

    useEffect(() => {
        if (debounceRef.current) clearTimeout(debounceRef.current)

        const text = input.trim()
        if (text.length < 3) { setSuggestions([]); return }

        debounceRef.current = setTimeout(async () => {
            try {
                const res = await fetch(apiUrl(`/api/agent/suggestions?q=${encodeURIComponent(text)}`))
                if (res.ok) {
                    const data = await res.json()
                    const items = data.suggestions || data.actions || []
                    setSuggestions(
                        items.slice(0, 3).map((item: unknown) => {
                            if (typeof item === 'string') {
                                return { label: item, action: item, panel: undefined }
                            }
                            if (item && typeof item === 'object') {
                                const obj = item as Record<string, unknown>
                                return {
                                    label: String(obj.label || obj.text || obj.title || 'Suggestion'),
                                    action: String(obj.action || obj.text || obj.title || ''),
                                    panel: typeof obj.panel === 'string' ? obj.panel : undefined,
                                }
                            }
                            return { label: String(item), action: String(item), panel: undefined }
                        }),
                    )
                } else {
                    setSuggestions(inferLocal(text))
                }
            } catch {
                setSuggestions(inferLocal(text))
            }
        }, 400)

        return () => { if (debounceRef.current) clearTimeout(debounceRef.current) }
    }, [input])

    const handleClick = useCallback((s: Suggestion) => {
        if (s.panel) onNavigate(s.panel, s.action)
        setSuggestions([])
    }, [onNavigate])

    if (suggestions.length === 0) return null

    return (
        <div className="suggestions-bar">
            {suggestions.map((s, i) => (
                <button key={i} className="btn btn--sm btn--pill" onClick={() => handleClick(s)}>
                    {s.label}
                </button>
            ))}
        </div>
    )
}

/* Local fallback when backend is offline */
function inferLocal(text: string): Suggestion[] {
    const lower = text.toLowerCase()
    const out: Suggestion[] = []

    const methods = ['did', 'difference-in-differences', 'iv ', 'instrumental', 'rdd', 'regression discontinuity', 'fixed effect', 'propensity', 'synthetic control', 'event study', 'ols']
    if (methods.some(m => lower.includes(m))) {
        out.push({ label: '→ Methods Lab', action: text, panel: 'methods' })
    }

    if (lower.includes('code') || lower.includes('stata') || lower.includes('regression') || lower.includes('estimate') || lower.includes('run ')) {
        out.push({ label: '→ Vibe Coder', action: text, panel: 'vibe' })
    }

    if (lower.includes('paper') || lower.includes('article') || lower.includes('study by') || lower.includes('read ')) {
        out.push({ label: '→ Paper Dive', action: text, panel: 'dive' })
    }

    if (lower.includes('find') || lower.includes('search') || lower.includes('look up') || lower.includes('literature on')) {
        out.push({ label: '→ Search', action: text, panel: 'search' })
    }

    if (lower.includes('cite') || lower.includes('citation') || lower.includes('reference')) {
        out.push({ label: '→ Citations', action: text, panel: 'citations' })
    }

    return out.slice(0, 3)
}
