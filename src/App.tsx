import { useEffect, useCallback, useState } from 'react'
import { useStore } from './store'
import PanelNav from './components/PanelNav'
import ErrorBoundary from './components/ErrorBoundary'
import WinniePanel from './panels/WinniePanel'
import LibraryPanel from './panels/LibraryPanel'
import SearchPanel from './panels/SearchPanel'
import VibeCoderPanel from './panels/VibeCoderPanel'
import MethodsLabPanel from './panels/MethodsLabPanel'
import PaperDivePanel from './panels/PaperDivePanel'
import CitationsPanel from './panels/CitationsPanel'
import { apiUrl } from './api'
import './styles/tokens.css'
import NotesDrawer from './components/NotesDrawer'
import './styles/shell.css'

const PANELS: Record<string, () => React.ReactNode> = {
    chat: () => <WinniePanel />,
    library: () => <LibraryPanel />,
    search: () => <SearchPanel />,
    vibe: () => <VibeCoderPanel />,
    methods: () => <MethodsLabPanel />,
    dive: () => <PaperDivePanel />,
    citations: () => <CitationsPanel />,
}

export default function App() {
    const { activeTab, setActiveTab, setBackendConnected } = useStore()
    const [dragOver, setDragOver] = useState(false)
    const [notesOpen, setNotesOpen] = useState(false)

    // Keyboard shortcuts: Cmd+1..7 to switch panels
    useEffect(() => {
        const TAB_KEYS: Record<string, string> = { '1': 'chat', '2': 'library', '3': 'search', '4': 'vibe', '5': 'methods', '6': 'dive', '7': 'citations' }
        const handler = (e: KeyboardEvent) => {
            if (!e.metaKey && !e.ctrlKey) return
            const tab = TAB_KEYS[e.key]
            if (tab && !(e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement)) {
                e.preventDefault()
                setActiveTab(tab as 'chat' | 'library' | 'search' | 'vibe' | 'methods' | 'dive' | 'citations')
            }
            if (e.key === 'n' && !(e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement)) {
                e.preventDefault()
                setNotesOpen(prev => !prev)
            }
        }
        window.addEventListener('keydown', handler)
        return () => window.removeEventListener('keydown', handler)
    }, [setActiveTab])

    // Health check polling
    useEffect(() => {
        const check = async () => {
            try {
                const res = await fetch(apiUrl('/api/status'))
                setBackendConnected(res.ok)
                const dot = document.getElementById('status-indicator')
                const label = dot?.nextElementSibling
                if (dot) dot.className = res.ok ? 'status-dot status-dot--online' : 'status-dot'
                if (label) label.textContent = res.ok ? 'Connected' : 'Offline'
            } catch {
                setBackendConnected(false)
            }
        }
        check()
        const iv = setInterval(check, 10000)
        return () => clearInterval(iv)
    }, [setBackendConnected])

    const handleDragOver = useCallback((e: React.DragEvent) => { e.preventDefault(); setDragOver(true) }, [])
    const handleDragLeave = useCallback(() => setDragOver(false), [])
    const handleDrop = useCallback(async (e: React.DragEvent) => {
        e.preventDefault()
        setDragOver(false)
        const files = Array.from(e.dataTransfer.files)
        if (files.length > 0) {
            try {
                for (const file of files) {
                    const form = new FormData()
                    form.append('file', file)
                    await fetch(apiUrl('/api/library/upload'), {
                        method: 'POST',
                        body: form,
                    })
                }
                await fetch(apiUrl('/api/index/run'), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ force: false }),
                })
            } catch { /* offline */ }
        }
    }, [])

    const renderPanel = PANELS[activeTab]

    return (
        <div className="app" onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}>
            <PanelNav />

            <main id="main-content" style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
                <ErrorBoundary key={activeTab}>
                    {renderPanel ? renderPanel() : (
                        <div className="placeholder">
                            <span className="placeholder__text">Panel not found</span>
                        </div>
                    )}
                </ErrorBoundary>
            </main>

            <NotesDrawer open={notesOpen} onClose={() => setNotesOpen(false)} />

            {dragOver && (
                <div style={{
                    position: 'fixed', inset: 0, background: 'rgba(0,113,227,0.08)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: 'var(--text-lg)', color: 'var(--accent)', fontWeight: 500,
                    zIndex: 999, backdropFilter: 'blur(4px)',
                }}>
                    Drop files to index
                </div>
            )}
        </div>
    )
}
