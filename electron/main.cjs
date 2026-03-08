const { app, BrowserWindow, shell } = require('electron')
const { spawn, spawnSync } = require('child_process')
const crypto = require('crypto')
const fs = require('fs')
const path = require('path')
const http = require('http')
const { pathToFileURL } = require('url')

/* ── E.D.I.T.H. M2 — Electron Main Process ────────────
   Minimal wrapper: opens a window, loads the Vite build.
   In dev: loads localhost:5174
   In prod: loads dist/index.html
   Auto-starts the Python backend if not already running.
   ─────────────────────────────────────────────────────── */

const isDev = !app.isPackaged
const BACKEND_PORT = Number(process.env.EDITH_M2_BACKEND_PORT || 8003)
const FRONTEND_PORT = Number(process.env.EDITH_M2_FRONTEND_PORT || 5176)
const HEALTH_PATH = '/api/status'
const LOCAL_BACKEND_ORIGIN = `http://127.0.0.1:${BACKEND_PORT}`
const M2_PROJECT_ROOT = path.resolve(__dirname, '..')
const EXTERNAL_M2_ROOT = path.join('/Volumes', 'Edith Bolt', 'Edith_M2')
const SHARED_DATA_ROOT = process.env.EDITH_SHARED_DATA_ROOT || path.join('/Volumes', 'Edith Bolt', 'Edith_M4')
const SHARED_CHROMA_DIR = process.env.EDITH_SHARED_CHROMA_DIR || path.join(SHARED_DATA_ROOT, 'ChromaDB')
const SHARED_COLLECTION = process.env.EDITH_SHARED_CHROMA_COLLECTION || 'edith_docs_pdf'
const BACKEND_ORIGIN = normalizeOrigin(
    process.env.EDITH_M2_BACKEND_URL || process.env.EDITH_BACKEND_URL,
    LOCAL_BACKEND_ORIGIN,
)
const HEALTH_CANDIDATES = Array.from(new Set([
    `${BACKEND_ORIGIN}${HEALTH_PATH}`,
    `${LOCAL_BACKEND_ORIGIN}${HEALTH_PATH}`,
]))
const USER_DATA_DIR = process.env.EDITH_M2_USER_DATA_DIR
    ? path.resolve(process.env.EDITH_M2_USER_DATA_DIR)
    : path.join(app.getPath('appData'), 'Edith_M2')
const SESSION_DATA_DIR = process.env.EDITH_M2_SESSION_DATA_DIR
    ? path.resolve(process.env.EDITH_M2_SESSION_DATA_DIR)
    : path.join(USER_DATA_DIR, 'SessionData')
let mainWindow = null
const BACKEND_WAIT_SECONDS = Math.max(30, Number(process.env.EDITH_BACKEND_WAIT_SECONDS || 120) || 120)
let backendProcess = null

function normalizeOrigin(raw, fallback = '') {
    const value = String(raw || fallback || '').trim()
    return value.replace(/\/+$/, '')
}

function pathExists(p) {
    try {
        fs.accessSync(p)
        return true
    } catch {
        return false
    }
}

function isFile(p) {
    try {
        return fs.statSync(p).isFile()
    } catch {
        return false
    }
}

function isExecutableFile(p) {
    try {
        return isFile(p) && fs.accessSync(p, fs.constants.X_OK) === undefined
    } catch {
        return false
    }
}

function isAllowedExternalUrl(rawUrl) {
    try {
        const parsed = new URL(rawUrl)
        if (parsed.protocol === 'https:' || parsed.protocol === 'mailto:') {
            return true
        }
        return parsed.protocol === 'http:' && (parsed.hostname === '127.0.0.1' || parsed.hostname === 'localhost')
    } catch {
        return false
    }
}

function configureStoragePathsEarly() {
    try {
        if (app.getPath('userData') !== USER_DATA_DIR) {
            app.setPath('userData', USER_DATA_DIR)
        }
        if (app.getPath('sessionData') !== SESSION_DATA_DIR) {
            app.setPath('sessionData', SESSION_DATA_DIR)
        }
        fs.mkdirSync(USER_DATA_DIR, { recursive: true })
        fs.mkdirSync(SESSION_DATA_DIR, { recursive: true })
    } catch (err) {
        console.warn('[edith-m2] Could not configure user/session paths:', err?.message || err)
    }
}

function resolvePythonCommand(backendRoot) {
    const pythonCandidates = [
        process.env.EDITH_PYTHON || '',
        path.join(backendRoot, '.venv', 'bin', 'python'),
        '/opt/homebrew/bin/python3',
        'python3',
    ].filter(Boolean)

    for (const cmd of pythonCandidates) {
        const probe = spawnSync(cmd, ['--version'], { stdio: 'ignore' })
        if (!probe.error && probe.status === 0) {
            return cmd
        }
    }
    return null
}

configureStoragePathsEarly()

function findBackendLaunchConfig() {
    const roots = [
        process.env.EDITH_M2_BACKEND_ROOT || '',
        process.env.EDITH_BACKEND_ROOT || '',
        app.isPackaged ? path.join(process.resourcesPath, 'edith_backend') : '',
        app.isPackaged ? path.join(process.resourcesPath, 'edith_backend', 'edith_backend') : '',
        path.join(M2_PROJECT_ROOT, 'electron', 'extraResources', 'edith_backend'),
        path.join('/Volumes', 'Edith Bolt', 'Edith_M2', 'electron', 'extraResources', 'edith_backend'),
        path.join('/Volumes', 'Edith Bolt', 'Edith_M4', 'electron', 'extraResources', 'edith_backend'),
        path.join('/Volumes', 'Edith Bolt', 'Edith_M4', 'dist', 'edith_backend'),
        path.join('/Volumes', 'Edith Bolt', 'Edith_M4', 'build', 'edith_backend'),
        path.join(process.env.HOME || '', 'Projects', 'edith_safe_chat'),
        '/Applications/Edith.app/Contents/Resources/edith_backend',
    ].filter(Boolean)

    for (const root of roots) {
        if (!pathExists(root)) continue

        const desktopLauncher = path.join(root, 'desktop_launcher.py')
        const serverMain = path.join(root, 'server', 'main.py')
        const pythonCmd = resolvePythonCommand(root)

        if (isFile(desktopLauncher) && pythonCmd) {
            return {
                backendRoot: root,
                command: pythonCmd,
                args: [desktopLauncher],
                mode: 'desktop_launcher.py',
            }
        }

        if (isFile(serverMain) && pythonCmd) {
            return {
                backendRoot: root,
                command: pythonCmd,
                args: ['-m', 'server.main'],
                mode: 'server.main',
            }
        }

        // Fallback to bundled binary only when Python launcher is unavailable.
        if (isExecutableFile(root)) {
            return {
                backendRoot: path.dirname(root),
                command: root,
                args: [],
                mode: 'binary',
            }
        }

        const bundledBinary = path.join(root, 'edith_backend')
        if (isExecutableFile(bundledBinary)) {
            return {
                backendRoot: root,
                command: bundledBinary,
                args: [],
                mode: 'binary',
            }
        }
    }
    return null
}

function probeHttp(url) {
    return new Promise((resolve) => {
        const req = http.get(url, (res) => {
            resolve(res.statusCode >= 200 && res.statusCode < 300)
        })
        req.on('error', () => resolve(false))
        req.setTimeout(2000, () => { req.destroy(); resolve(false) })
    })
}

// ── Check if backend is already running ────────────────
async function isBackendRunning() {
    for (const candidate of HEALTH_CANDIDATES) {
        if (await probeHttp(candidate)) {
            return true
        }
    }
    return false
}

// ── Start backend if not running ───────────────────────
async function ensureBackend() {
    if (await isBackendRunning()) {
        console.log('[edith-m2] Backend already running on :' + BACKEND_PORT)
        return true
    }

    const launch = findBackendLaunchConfig()
    if (!launch) {
        console.error('[edith-m2] No usable backend found. Set EDITH_BACKEND_ROOT to a folder with desktop_launcher.py or server/main.py')
        return false
    }

    const appStateDir =
        process.env.EDITH_M2_APP_DATA_DIR ||
        path.join(app.getPath('userData'), 'backend-state')
    const packagedDataRoot =
        path.basename(M2_PROJECT_ROOT) === 'app.asar' && pathExists(EXTERNAL_M2_ROOT)
            ? EXTERNAL_M2_ROOT
            : M2_PROJECT_ROOT
    const docsRoot =
        process.env.EDITH_M2_DATA_ROOT ||
        packagedDataRoot
    const chromaDir =
        process.env.EDITH_M2_CHROMA_DIR ||
        SHARED_CHROMA_DIR
    const collectionName =
        process.env.EDITH_M2_CHROMA_COLLECTION ||
        SHARED_COLLECTION
    const collectionStrict =
        process.env.EDITH_M2_CHROMA_COLLECTION_STRICT ||
        'true'
    const requireSharedIndex = (process.env.EDITH_M2_REQUIRE_SHARED_INDEX || 'true').toLowerCase() === 'true'
    const sharedSqlite = path.join(chromaDir, 'chroma.sqlite3')
    const sharedDotenv =
        process.env.EDITH_M2_DOTENV_PATH ||
        (isFile(path.join(SHARED_DATA_ROOT, '.env')) ? path.join(SHARED_DATA_ROOT, '.env') : '')

    try {
        fs.mkdirSync(appStateDir, { recursive: true })
        if (!requireSharedIndex) {
            fs.mkdirSync(chromaDir, { recursive: true })
        }
    } catch (err) {
        console.warn('[edith-m2] Could not create backend data dir:', err?.message || err)
    }

    if (requireSharedIndex && !isFile(sharedSqlite)) {
        console.error(`[edith-m2] Shared index required but missing: ${sharedSqlite}`)
        console.error('[edith-m2] Set EDITH_M2_REQUIRE_SHARED_INDEX=false to allow boot without shared index.')
        return false
    }

    const sessionToken = process.env.EDITH_SESSION_TOKEN || crypto.randomBytes(24).toString('hex')

    console.log(`[edith-m2] Starting backend (${launch.mode}) in: ${launch.backendRoot}`)
    console.log(
        `[edith-m2] Runtime isolation docs=${docsRoot} state=${appStateDir} chroma=${chromaDir} collection=${collectionName} strict=${collectionStrict}`,
    )
    backendProcess = spawn(launch.command, launch.args, {
        cwd: launch.backendRoot,
        env: {
            ...process.env,
            EDITH_PORT: String(BACKEND_PORT),
            EDITH_WORKERS: '1',           // Force single worker — multi-worker crash-loops on macOS
            EDITH_OPEN_BROWSER: 'false',
            EDITH_DESKTOP_MODE: 'electron',
            EDITH_DATA_ROOT: docsRoot,
            EDITH_APP_DATA_DIR: appStateDir,
            EDITH_CHROMA_DIR: chromaDir,
            EDITH_CHROMA_COLLECTION: collectionName,
            EDITH_CHROMA_COLLECTION_STRICT: String(collectionStrict),
            EDITH_DOTENV_PATH: sharedDotenv,
            EDITH_USER_DATA_DIR: USER_DATA_DIR,
            EDITH_SESSION_DATA_DIR: SESSION_DATA_DIR,
            EDITH_SESSION_TOKEN: sessionToken,
            MallocStackLogging: '',        // Suppress MallocStackLogging conflict
        },
        stdio: ['ignore', 'pipe', 'pipe'],
    })

    backendProcess.stdout.on('data', (d) => console.log('[backend]', d.toString().trim()))
    backendProcess.stderr.on('data', (d) => console.error('[backend]', d.toString().trim()))
    backendProcess.on('exit', (code) => {
        console.log(`[edith-m2] Backend exited with code ${code}`)
        backendProcess = null
    })

    // Wait for backend health (default 120s, configurable via EDITH_BACKEND_WAIT_SECONDS).
    for (let i = 0; i < BACKEND_WAIT_SECONDS; i++) {
        await new Promise((r) => setTimeout(r, 1000))
        if (await isBackendRunning()) {
            console.log('[edith-m2] Backend is healthy')
            return true
        }
    }
    console.warn(`[edith-m2] Backend did not become healthy in ${BACKEND_WAIT_SECONDS}s`)
    return false
}

// ── Create the main window ─────────────────────────────
function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        minWidth: 900,
        minHeight: 600,
        title: 'E.D.I.T.H. M2',
        titleBarStyle: 'hiddenInset',
        backgroundColor: '#0f1729',
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
        },
    })

    // Load the app and pin backend origin for file:// packaged mode.
    if (isDev) {
        mainWindow.loadURL(`http://localhost:${FRONTEND_PORT}?backend=${encodeURIComponent(BACKEND_ORIGIN)}`)
    } else {
        const indexFile = path.join(__dirname, '..', 'dist', 'index.html')
        const indexUrl = pathToFileURL(indexFile)
        indexUrl.searchParams.set('backend', BACKEND_ORIGIN)
        mainWindow.loadURL(indexUrl.toString())
    }

    // Open external links in browser
    mainWindow.webContents.setWindowOpenHandler(({ url }) => {
        if (isAllowedExternalUrl(url)) {
            shell.openExternal(url)
        }
        return { action: 'deny' }
    })
    mainWindow.webContents.on('will-navigate', (event, url) => {
        const isInternal = isDev
            ? url.startsWith(`http://localhost:${FRONTEND_PORT}`)
            : url.startsWith('file://')
        if (isInternal) return
        event.preventDefault()
        if (isAllowedExternalUrl(url)) {
            shell.openExternal(url)
        }
    })

    mainWindow.webContents.on('did-fail-load', (_event, code, description, url, isMainFrame) => {
        if (!isMainFrame) return
        console.error(`[edith-m2] did-fail-load code=${code} reason=${description} url=${url}`)
    })
    mainWindow.webContents.on('render-process-gone', (_event, details) => {
        console.error(`[edith-m2] Renderer process gone: ${details.reason}`)
    })

    mainWindow.on('closed', () => { mainWindow = null })
}

// ── App lifecycle ──────────────────────────────────────
app.whenReady().then(async () => {
    // Start backend first to avoid file:// frontend fetch races during cold boot.
    try {
        await ensureBackend()
    } catch (err) {
        console.error('[edith-m2] Backend startup failed:', err?.message || err)
    }

    createWindow()

    app.on('activate', async () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            try {
                await ensureBackend()
            } catch (err) {
                console.error('[edith-m2] Backend startup failed on activate:', err?.message || err)
            }
            createWindow()
        }
    })
})

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', () => {
    if (backendProcess) {
        console.log('[edith-m2] Stopping backend...')
        backendProcess.kill('SIGTERM')
    }
})
