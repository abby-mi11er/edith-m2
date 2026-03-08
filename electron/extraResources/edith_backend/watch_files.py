import os
import sys
import time
import subprocess
from pathlib import Path
from threading import Timer
from dotenv import load_dotenv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

dotenv_override = os.environ.get("EDITH_DOTENV_PATH")
if dotenv_override:
    load_dotenv(dotenv_path=Path(dotenv_override).expanduser(), override=False)
else:
    load_dotenv()

ROOT = Path(os.environ.get("EDITH_DATA_ROOT", "")).expanduser()
DEBOUNCE = float(os.environ.get("EDITH_WATCH_DEBOUNCE", "2.0"))
RETRIEVAL_BACKEND = os.environ.get("EDITH_RETRIEVAL_BACKEND", "google").strip().lower()

if not ROOT or not ROOT.exists():
    raise SystemExit("EDITH_DATA_ROOT is missing or invalid.")

VALID_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".txt", ".md", ".rtf", ".odt", ".tex",
    ".csv", ".tsv", ".xlsx", ".xls", ".json", ".jsonl",
    ".py", ".ipynb", ".js", ".ts", ".sql", ".r", ".R",
}

IGNORE_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__"}


def run_index():
    script_name = "index_files.py" if RETRIEVAL_BACKEND == "google" else "chroma_index.py"
    script = Path(__file__).parent / script_name
    if not script.exists():
        return
    subprocess.run([sys.executable, str(script)], capture_output=False)


class DebouncedHandler(FileSystemEventHandler):
    def __init__(self):
        self.timer = None
        self.is_indexing = False

    def on_any_event(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if any(part in IGNORE_DIRS for part in path.parts):
            return
        if path.suffix.lower() not in VALID_EXTENSIONS:
            return
        if self.timer:
            self.timer.cancel()
        self.timer = Timer(DEBOUNCE, self.trigger)
        self.timer.start()

    def trigger(self):
        if self.is_indexing:
            return
        self.is_indexing = True
        try:
            run_index()
        finally:
            self.is_indexing = False


if __name__ == "__main__":
    handler = DebouncedHandler()
    observer = Observer()
    observer.schedule(handler, str(ROOT), recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
