import { useState, useEffect, useCallback, useRef } from 'react'
import { apiUrl } from '../api'

/* ── Notes Drawer ──────────────────────────────────── */

const STORAGE_KEY = 'edith_m2_notes'

interface Note {
    id: string
    text: string
    ts: number
}

function loadNotes(): Note[] {
    try {
        const raw = localStorage.getItem(STORAGE_KEY)
        if (!raw) return []
        const parsed = JSON.parse(raw)
        return Array.isArray(parsed) ? parsed.filter((n: any) => n?.id && n?.text) : []
    } catch { return [] }
}

function saveNotes(notes: Note[]) {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(notes)) } catch { /* full */ }
}

export default function NotesDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
    const [notes, setNotes] = useState<Note[]>(loadNotes)
    const [draft, setDraft] = useState('')
    const inputRef = useRef<HTMLTextAreaElement>(null)

    useEffect(() => { if (open) inputRef.current?.focus() }, [open])

    const addNote = useCallback(() => {
        const text = draft.trim()
        if (!text) return
        const next = [{ id: crypto.randomUUID(), text, ts: Date.now() }, ...notes]
        setNotes(next)
        saveNotes(next)
        setDraft('')

        // Optional: sync to backend (fire-and-forget)
        fetch(apiUrl('/api/notes'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content: text }),
        }).catch(() => {})
    }, [draft, notes])

    const deleteNote = useCallback((id: string) => {
        const next = notes.filter(n => n.id !== id)
        setNotes(next)
        saveNotes(next)
    }, [notes])

    if (!open) return null

    return (
        <div className="notes-drawer-overlay" onClick={onClose}>
            <div className="notes-drawer" onClick={e => e.stopPropagation()}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-4)' }}>
                    <span style={{ fontWeight: 600, fontSize: 'var(--text-base)' }}>Quick Notes</span>
                    <button className="btn btn--sm btn--ghost" onClick={onClose}>✕</button>
                </div>

                <div style={{ display: 'flex', gap: 'var(--space-2)', marginBottom: 'var(--space-4)' }}>
                    <textarea
                        ref={inputRef}
                        className="input"
                        value={draft}
                        onChange={e => setDraft(e.target.value)}
                        onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); addNote() } }}
                        placeholder="Jot something down..."
                        rows={2}
                        style={{ resize: 'none', flex: 1 }}
                    />
                    <button className="btn btn--primary btn--sm" onClick={addNote} disabled={!draft.trim()}
                        style={{ alignSelf: 'flex-end' }}>
                        Add
                    </button>
                </div>

                <div style={{ flex: 1, overflowY: 'auto' }}>
                    {notes.length === 0 && (
                        <div className="placeholder">
                            <span className="placeholder__text">No notes yet. Cmd+N anytime to jot things down.</span>
                        </div>
                    )}
                    {notes.map(n => (
                        <div key={n.id} className="card" style={{ marginBottom: 'var(--space-2)', position: 'relative' }}>
                            <div style={{ fontSize: 'var(--text-sm)', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>{n.text}</div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 'var(--space-2)' }}>
                                <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-tertiary)' }}>
                                    {new Date(n.ts).toLocaleString()}
                                </span>
                                <button className="btn btn--sm btn--ghost" onClick={() => deleteNote(n.id)}
                                    style={{ fontSize: 'var(--text-xs)', opacity: 0.5 }}>
                                    Delete
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}
