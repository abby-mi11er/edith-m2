import { useState, useRef, useEffect, useCallback } from 'react'
import { useStore } from '../store'
import { apiUrl } from '../api'

/* ── Panel Navigation ──────────────────────────────── */

interface NavItem { id: string; label: string }
interface NavSection { section: string; items: NavItem[] }

const SECTIONS: NavSection[] = [
    {
        section: 'Research', items: [
            { id: 'chat', label: 'Winnie' },
            { id: 'library', label: 'Library' },
            { id: 'search', label: 'Search' },
        ]
    },
    {
        section: 'Tools', items: [
            { id: 'vibe', label: 'Vibe Coder' },
            { id: 'methods', label: 'Methods Lab' },
            { id: 'dive', label: 'Paper Dive' },
            { id: 'citations', label: 'Citations' },
        ]
    },
]

interface DashboardData {
    status: string
    subsystems: Record<string, { status: string; detail: string }>
    uptime_seconds: number
}

export default function PanelNav() {
    const { activeTab, setActiveTab } = useStore()
    const [openSection, setOpenSection] = useState<string | null>(null)
    const [showDashboard, setShowDashboard] = useState(false)
    const [dashboard, setDashboard] = useState<DashboardData | null>(null)
    const navRef = useRef<HTMLDivElement>(null)

    const activeSection = SECTIONS.find(s =>
        s.items.some(i => i.id === activeTab)
    )?.section || 'Research'

    useEffect(() => {
        const handler = (e: MouseEvent) => {
            if (navRef.current && !navRef.current.contains(e.target as Node)) {
                setOpenSection(null)
            }
        }
        document.addEventListener('mousedown', handler)
        return () => document.removeEventListener('mousedown', handler)
    }, [])

    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            if (e.key === 'Escape') { setOpenSection(null); setShowDashboard(false) }
        }
        window.addEventListener('keydown', handler)
        return () => window.removeEventListener('keydown', handler)
    }, [])

    const handleSelect = useCallback((id: string) => {
        setActiveTab(id as any)
        setOpenSection(null)
    }, [setActiveTab])

    const toggleDashboard = useCallback(async () => {
        if (showDashboard) { setShowDashboard(false); return }
        setShowDashboard(true)
        try {
            const res = await fetch(apiUrl('/api/status'))
            if (res.ok) setDashboard(await res.json())
        } catch { /* offline */ }
    }, [showDashboard])

    const currentLabel = SECTIONS.flatMap(s => s.items).find(i => i.id === activeTab)?.label || 'Winnie'

    return (
        <div className="titlebar">
            <div className="titlebar__brand">E.D.I.T.H.  /  {currentLabel}</div>

            <nav className="nav" ref={navRef} aria-label="Panel navigation" style={{ border: 'none', padding: 0 }}>
                {SECTIONS.map(section => {
                    const isActive = section.section === activeSection
                    const isOpen = openSection === section.section

                    return (
                        <div key={section.section} className="nav__section">
                            <button
                                className={`nav__section-label ${isActive ? 'active' : ''}`}
                                onClick={() => setOpenSection(isOpen ? null : section.section)}
                                aria-expanded={isOpen}
                                style={isActive ? { color: 'var(--accent)' } : {}}
                            >
                                {section.section}
                                <svg width="8" height="5" viewBox="0 0 8 5" fill="none" style={{
                                    transform: isOpen ? 'rotate(180deg)' : 'none',
                                    transition: 'transform 200ms',
                                }}>
                                    <path d="M1 1L4 4L7 1" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                                </svg>
                            </button>

                            {isOpen && (
                                <div className="nav__dropdown">
                                    {section.items.map(item => (
                                        <button
                                            key={item.id}
                                            className={`nav__item ${activeTab === item.id ? 'nav__item--active' : ''}`}
                                            onClick={() => handleSelect(item.id)}
                                        >
                                            {item.label}
                                        </button>
                                    ))}
                                </div>
                            )}
                        </div>
                    )
                })}
            </nav>

            <div className="titlebar__status" style={{ cursor: 'pointer', position: 'relative' }}
                onClick={toggleDashboard}>
                <span className="status-dot" id="status-indicator" />
                <span>Offline</span>

                {showDashboard && dashboard && (
                    <div style={{
                        position: 'absolute', top: '100%', right: 0, marginTop: 8,
                        background: 'var(--surface)', border: '1px solid var(--border)',
                        borderRadius: 'var(--radius-lg)', padding: 'var(--space-4)',
                        boxShadow: 'var(--shadow-lg)', minWidth: 260, zIndex: 100,
                        fontSize: 'var(--text-sm)', color: 'var(--text-secondary)',
                    }} onClick={e => e.stopPropagation()}>
                        <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: 'var(--space-3)' }}>
                            System Dashboard
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                            <div>Uptime: {Math.floor(dashboard.uptime_seconds / 60)}m</div>
                            {Object.entries(dashboard.subsystems || {}).map(([key, val]) => (
                                <div key={key} style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
                                    <span style={{ textTransform: 'capitalize' }}>{key}</span>
                                    <span style={{ color: val.status === 'ok' ? '#34c759' : 'var(--text-tertiary)' }}>
                                        {val.detail}
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}
