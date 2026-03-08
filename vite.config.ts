import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const FRONTEND_PORT = Number(process.env.EDITH_M2_FRONTEND_PORT || 5176)
const BACKEND_PORT = Number(process.env.EDITH_M2_BACKEND_PORT || 8003)

export default defineConfig({
  // Required for Electron packaged `file://` loads.
  base: './',
  plugins: [react()],
  server: {
    port: FRONTEND_PORT,
    strictPort: true,
    headers: {
      'X-Content-Type-Options': 'nosniff',
      'X-Frame-Options': 'DENY',
      'Referrer-Policy': 'strict-origin-when-cross-origin',
      'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
    },
    proxy: {
      '/api': {
        target: `http://127.0.0.1:${BACKEND_PORT}`,
        changeOrigin: true,
      },
      '/chat': {
        target: `http://127.0.0.1:${BACKEND_PORT}`,
        changeOrigin: true,
      },
    },
  },
  build: {
    sourcemap: false,
  },
})
