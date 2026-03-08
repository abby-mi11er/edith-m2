import { useStore } from '../store'
import type { SavedChat } from '../store'

function groupByDate(chats: SavedChat[]): Record<string, SavedChat[]> {
    const now = Date.now()
    const DAY = 86_400_000
    const groups: Record<string, SavedChat[]> = {}
    for (const chat of chats) {
        const age = now - chat.updatedAt
        const label = age < DAY ? 'Today'
            : age < DAY * 2 ? 'Yesterday'
                : age < DAY * 7 ? 'This Week'
                    : 'Older'
            ; (groups[label] ||= []).push(chat)
    }
    return groups
}

function formatTime(ts: number): string {
    const d = new Date(ts)
    const now = new Date()
    if (d.toDateString() === now.toDateString()) {
        return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
    return d.toLocaleDateString([], { month: 'short', day: 'numeric' })
}

export default function ChatHistory() {
    const { savedChats, activeChatId, historyOpen, loadChat, deleteChat, newChat } = useStore()

    if (!historyOpen) return null

    const groups = groupByDate(savedChats)
    const order = ['Today', 'Yesterday', 'This Week', 'Older']

    return (
        <aside className="chat-history">
            <div className="chat-history__header">
                <span className="chat-history__title">History</span>
                <button className="chat-history__new" onClick={newChat} title="New chat">
                    + New Chat
                </button>
            </div>

            <div className="chat-history__list">
                {savedChats.length === 0 && (
                    <div className="chat-history__empty">No saved conversations yet</div>
                )}
                {order.map(label => {
                    const items = groups[label]
                    if (!items?.length) return null
                    return (
                        <div key={label} className="chat-history__group">
                            <div className="chat-history__group-label">{label}</div>
                            {items.map(chat => (
                                <button
                                    key={chat.id}
                                    className={`chat-history__item ${chat.id === activeChatId ? 'chat-history__item--active' : ''}`}
                                    onClick={() => loadChat(chat.id)}
                                    title={chat.title}
                                >
                                    <span className="chat-history__item-title">{chat.title}</span>
                                    <span className="chat-history__item-meta">
                                        <span className="chat-history__item-time">{formatTime(chat.updatedAt)}</span>
                                        <span
                                            className="chat-history__item-delete"
                                            onClick={(e) => { e.stopPropagation(); deleteChat(chat.id) }}
                                            title="Delete"
                                        >×</span>
                                    </span>
                                </button>
                            ))}
                        </div>
                    )
                })}
            </div>
        </aside>
    )
}
