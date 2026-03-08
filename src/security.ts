/* ── Security utilities ────────────────────────────── */

/**
 * Sanitize user text input: strip HTML tags, limit length, trim.
 * Prevents XSS via crafted input that could end up in DOM.
 */
export function sanitize(input: string, maxLength = 10000): string {
    return input
        .replace(/<[^>]*>/g, '')       // strip HTML tags
        .replace(/javascript:/gi, '')   // strip JS protocol
        .replace(/on\w+\s*=/gi, '')     // strip event handlers
        .slice(0, maxLength)
        .trim()
}

/**
 * Validate a URL to prevent open redirect / SSRF via user input.
 * Only allows http/https protocols.
 */
export function isValidUrl(url: string): boolean {
    try {
        const parsed = new URL(url)
        return ['http:', 'https:'].includes(parsed.protocol)
    } catch {
        return false
    }
}

/**
 * Wrap a fetch call with timeout to prevent hanging requests.
 */
export async function safeFetch(
    url: string,
    options?: RequestInit,
    timeoutMs = 30000
): Promise<Response> {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), timeoutMs)
    try {
        return await fetch(url, { ...options, signal: controller.signal })
    } finally {
        clearTimeout(timer)
    }
}
