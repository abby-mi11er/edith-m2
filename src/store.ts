import { create } from 'zustand'

/* ── Types ──────────────────────────── */

export interface SourceRef {
    title: string
    author?: string
    year?: string
    file_name?: string
    page?: number
    [key: string]: any
}

export interface ChatMessage {
    role: 'user' | 'assistant'
    content: string
    sources?: SourceRef[]
}

export interface SavedChat {
    id: string
    title: string
    messages: ChatMessage[]
    mode: string
    createdAt: number
    updatedAt: number
}

type TabId = 'chat' | 'library' | 'search' | 'vibe' | 'methods' | 'dive' | 'citations'

interface CodeMessage { role: 'user' | 'assistant'; content: string; language?: string }

// Persist chat to localStorage
const STORAGE_KEY = 'edith_m2_chat'

function storageGet(key: string): string | null {
    try { return localStorage.getItem(key) } catch { return null }
}

function storageSet(key: string, value: string) {
    try { localStorage.setItem(key, value) } catch { /* unavailable/full */ }
}

function storageRemove(key: string) {
    try { localStorage.removeItem(key) } catch { /* unavailable */ }
}

// Persist chat history
const HISTORY_KEY = 'edith_m2_history'
const ACTIVE_CHAT_KEY = 'edith_m2_active_chat'

function loadHistory(): SavedChat[] {
    try {
        const raw = storageGet(HISTORY_KEY)
        if (!raw) return []
        const parsed = JSON.parse(raw)
        return Array.isArray(parsed) ? parsed : []
    } catch { return [] }
}

function saveHistory(chats: SavedChat[]) {
    storageSet(HISTORY_KEY, JSON.stringify(chats.slice(0, 50)))
}

function generateTitle(messages: ChatMessage[]): string {
    const first = messages.find(m => m.role === 'user')
    if (!first) return 'New Chat'
    const text = first.content.replace(/\s+/g, ' ').trim()
    return text.length > 50 ? text.slice(0, 47) + '...' : text
}

function loadMessages(): ChatMessage[] {
    try {
        const raw = storageGet(STORAGE_KEY)
        if (!raw) return []
        const parsed = JSON.parse(raw)
        if (!Array.isArray(parsed)) { storageRemove(STORAGE_KEY); return [] }
        // Validate each entry — drop corrupted messages
        const valid = parsed.filter((m: any) =>
            m && typeof m === 'object' &&
            (m.role === 'user' || m.role === 'assistant') &&
            typeof m.content === 'string' && m.content.length > 0
        ).map((m: any) => ({
            role: m.role as 'user' | 'assistant',
            content: String(m.content),
            sources: Array.isArray(m.sources) ? m.sources.map((s: any) =>
                typeof s === 'string' ? { title: s } : s
            ).filter(Boolean) : [],
        }))
        if (valid.length !== parsed.length) {
            // Had corrupted entries — save cleaned version
            storageSet(STORAGE_KEY, JSON.stringify(valid.slice(-100)))
        }
        return valid
    } catch {
        return []
    }
}

function saveMessages(msgs: ChatMessage[]) {
    storageSet(STORAGE_KEY, JSON.stringify(msgs.slice(-100)))
}
let _saveTimer: ReturnType<typeof setTimeout> | null = null
function saveMessagesThrottled(msgs: ChatMessage[]) {
    if (_saveTimer) clearTimeout(_saveTimer)
    _saveTimer = setTimeout(() => saveMessages(msgs), 500)
}

interface EdithStore {
    /* Navigation */
    activeTab: TabId
    setActiveTab: (tab: TabId) => void

    /* Chat */
    messages: ChatMessage[]
    addMessage: (msg: { role: 'user' | 'assistant'; content: string }) => void
    updateLastMessage: (content: string, sources?: SourceRef[]) => void
    clearMessages: () => void
    isStreaming: boolean
    setStreaming: (s: boolean) => void
    committeeMode: boolean
    setCommitteeMode: (c: boolean) => void

    /* Chat History */
    savedChats: SavedChat[]
    activeChatId: string | null
    historyOpen: boolean
    saveCurrentChat: (mode?: string) => void
    loadChat: (id: string) => void
    deleteChat: (id: string) => void
    newChat: () => void
    toggleHistory: () => void

    /* Backend */
    backendConnected: boolean
    setBackendConnected: (c: boolean) => void

    /* Vibe Coder — persisted to sessionStorage */
    vibeMessages: CodeMessage[]
    vibeLang: 'stata' | 'r' | 'python'
    setVibeMessages: (msgs: CodeMessage[]) => void
    addVibeMessage: (msg: CodeMessage) => void
    setVibeLang: (lang: 'stata' | 'r' | 'python') => void
    clearVibeMessages: () => void
}

export const useStore = create<EdithStore>((set) => ({
    /* Navigation */
    activeTab: 'chat',
    setActiveTab: (tab) => set({ activeTab: tab }),

    /* Chat — persisted to localStorage */
    messages: loadMessages(),
    addMessage: (msg) => set((s) => {
        const msgs = [...s.messages, { ...msg, sources: [] }]
        saveMessages(msgs)
        return { messages: msgs }
    }),
    updateLastMessage: (content, sources) => set((s) => {
        const msgs = [...s.messages]
        const last = msgs[msgs.length - 1]
        if (last && last.role === 'assistant') {
            // Preserve rich source objects for author-date rendering
            const safeSources = sources
                ? sources.map((src: any) =>
                    typeof src === 'string' ? { title: src } as SourceRef : (src as SourceRef)
                ).filter(Boolean)
                : last.sources
            msgs[msgs.length - 1] = { ...last, content, sources: safeSources }
        }
        saveMessagesThrottled(msgs)
        return { messages: msgs }
    }),
    clearMessages: () => {
        // Save current chat before clearing (if non-empty)
        const state = useStore.getState()
        if (state.messages.length > 0) {
            state.saveCurrentChat()
        }
        storageRemove(STORAGE_KEY)
        storageRemove(ACTIVE_CHAT_KEY)
        set({ messages: [], activeChatId: null })
    },
    isStreaming: false,
    setStreaming: (s) => set({ isStreaming: s }),
    committeeMode: false,
    setCommitteeMode: (c) => set({ committeeMode: c }),

    /* Chat History */
    savedChats: loadHistory(),
    activeChatId: storageGet(ACTIVE_CHAT_KEY) || null,
    historyOpen: false,
    saveCurrentChat: (mode = 'grounded') => set((s) => {
        if (s.messages.length === 0) return s
        const now = Date.now()
        const existing = s.activeChatId ? s.savedChats.find(c => c.id === s.activeChatId) : null
        if (existing) {
            const chats = s.savedChats.map(c => c.id === existing.id
                ? { ...c, messages: [...s.messages], title: generateTitle(s.messages), updatedAt: now }
                : c
            )
            saveHistory(chats)
            return { savedChats: chats }
        }
        const id = `chat_${now}_${Math.random().toString(36).slice(2, 8)}`
        const newChat: SavedChat = {
            id, title: generateTitle(s.messages),
            messages: [...s.messages], mode,
            createdAt: now, updatedAt: now,
        }
        const chats = [newChat, ...s.savedChats]
        storageSet(ACTIVE_CHAT_KEY, id)
        saveHistory(chats)
        return { savedChats: chats, activeChatId: id }
    }),
    loadChat: (id) => set((s) => {
        const chat = s.savedChats.find(c => c.id === id)
        if (!chat) return s
        saveMessages(chat.messages)
        storageSet(ACTIVE_CHAT_KEY, id)
        return { messages: [...chat.messages], activeChatId: id }
    }),
    deleteChat: (id) => set((s) => {
        const chats = s.savedChats.filter(c => c.id !== id)
        saveHistory(chats)
        if (s.activeChatId === id) {
            storageRemove(STORAGE_KEY)
            storageRemove(ACTIVE_CHAT_KEY)
            return { savedChats: chats, activeChatId: null, messages: [] }
        }
        return { savedChats: chats }
    }),
    newChat: () => set((s) => {
        if (s.messages.length > 0) {
            useStore.getState().saveCurrentChat()
        }
        storageRemove(STORAGE_KEY)
        storageRemove(ACTIVE_CHAT_KEY)
        return { messages: [], activeChatId: null }
    }),
    toggleHistory: () => set((s) => ({ historyOpen: !s.historyOpen })),

    /* Backend */
    backendConnected: false,
    setBackendConnected: (c) => set({ backendConnected: c }),

    /* Vibe Coder — persisted to sessionStorage */
    vibeMessages: (() => {
        try {
            const raw = sessionStorage.getItem('edith_m2_vibe')
            return raw ? JSON.parse(raw) : []
        } catch { return [] }
    })(),
    vibeLang: (() => {
        try {
            return (sessionStorage.getItem('edith_m2_vibe_lang') as 'stata' | 'r' | 'python') || 'stata'
        } catch { return 'stata' as const }
    })(),
    setVibeMessages: (msgs) => {
        try { sessionStorage.setItem('edith_m2_vibe', JSON.stringify(msgs.slice(-50))) } catch { }
        set({ vibeMessages: msgs })
    },
    addVibeMessage: (msg) => set((s) => {
        const msgs = [...s.vibeMessages, msg]
        try { sessionStorage.setItem('edith_m2_vibe', JSON.stringify(msgs.slice(-50))) } catch { }
        return { vibeMessages: msgs }
    }),
    setVibeLang: (lang) => {
        try { sessionStorage.setItem('edith_m2_vibe_lang', lang) } catch { }
        set({ vibeLang: lang })
    },
    clearVibeMessages: () => {
        try { sessionStorage.removeItem('edith_m2_vibe') } catch { }
        set({ vibeMessages: [] })
    },
}))
