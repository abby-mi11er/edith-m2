import { useState, useEffect } from 'react'
import { apiUrl } from '../api'

/* ── Proactive Suggestions Hook ────────────────────── */

export interface ProactiveSuggestion {
    text: string
    action?: string
}

export function useSuggestions(context: string, limit = 3): ProactiveSuggestion[] {
    const [suggestions, setSuggestions] = useState<ProactiveSuggestion[]>([])

    useEffect(() => {
        if (!context || context.length < 5) { setSuggestions([]); return }

        let cancelled = false
        const timer = setTimeout(async () => {
            try {
                const res = await fetch(apiUrl(`/api/agent/suggestions?q=${encodeURIComponent(context.slice(0, 500))}`))
                if (res.ok && !cancelled) {
                    const data = await res.json()
                    const items = data.suggestions || data.results || []
                    setSuggestions(
                        items.slice(0, limit).map((s: any) =>
                            typeof s === 'string'
                                ? { text: s }
                                : { text: s.text || s.title || s.suggestion || String(s), action: s.action },
                        ),
                    )
                }
            } catch { /* offline — no suggestions */ }
        }, 600)

        return () => { cancelled = true; clearTimeout(timer) }
    }, [context, limit])

    return suggestions
}

/* Presentational mini-bar */
export function SuggestionChips({
    suggestions,
    onSelect,
}: {
    suggestions: ProactiveSuggestion[]
    onSelect: (text: string) => void
}) {
    if (suggestions.length === 0) return null

    return (
        <div className="suggestions-bar">
            {suggestions.map((s, i) => (
                <button
                    key={i}
                    className="btn btn--sm btn--pill"
                    onClick={() => onSelect(s.text)}
                >
                    {s.text.length > 60 ? `${s.text.slice(0, 57)}...` : s.text}
                </button>
            ))}
        </div>
    )
}
