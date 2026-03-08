const FALLBACK_BACKEND_PORT = 8003

function normalizePath(pathname: string): string {
  if (!pathname) return '/'
  if (pathname.startsWith('http://') || pathname.startsWith('https://')) return pathname
  return pathname.startsWith('/') ? pathname : `/${pathname}`
}

function backendFromQuery(): string {
  try {
    const value = new URLSearchParams(window.location.search).get('backend') || ''
    return value.replace(/\/+$/, '')
  } catch {
    return ''
  }
}

function inferBackendOrigin(): string {
  const queryOrigin = backendFromQuery()
  if (queryOrigin) return queryOrigin

  if (window.location.protocol === 'http:' || window.location.protocol === 'https:') {
    return ''
  }

  return `http://127.0.0.1:${FALLBACK_BACKEND_PORT}`
}

const BACKEND_ORIGIN = inferBackendOrigin()

export function apiUrl(pathname: string): string {
  const normalized = normalizePath(pathname)
  if (normalized.startsWith('http://') || normalized.startsWith('https://')) {
    return normalized
  }
  return BACKEND_ORIGIN ? `${BACKEND_ORIGIN}${normalized}` : normalized
}

export function backendUrl(pathname: string): string {
  return apiUrl(pathname)
}

