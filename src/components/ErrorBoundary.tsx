import { Component, type ReactNode } from 'react'

/* ── Error Boundary ────────────────────────────────── */

interface Props { children: ReactNode }
interface State { hasError: boolean }

export default class ErrorBoundary extends Component<Props, State> {
    state: State = { hasError: false }

    static getDerivedStateFromError(): State {
        return { hasError: true }
    }

    componentDidCatch(error: Error) {
        // Log to console only — never expose stack traces to the UI
        console.error('[E.D.I.T.H.] Panel error:', error.message)
    }

    render() {
        if (this.state.hasError) {
            return (
                <div className="panel" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <div className="placeholder">
                        <span className="placeholder__title">Something went wrong</span>
                        <span className="placeholder__text">
                            Try refreshing the page. If the issue persists, restart the backend.
                        </span>
                        <button className="btn btn--primary btn--sm" style={{ marginTop: 12 }}
                            onClick={() => this.setState({ hasError: false })}>
                            Retry
                        </button>
                    </div>
                </div>
            )
        }
        return this.props.children
    }
}
