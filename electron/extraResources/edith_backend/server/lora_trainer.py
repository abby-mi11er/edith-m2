"""
LoRA Trainer — Local Fine-Tuning with MLX
============================================
On-device LoRA sharpening using Apple's MLX framework.
Takes corrections from autolearn.jsonl + feedback.sqlite3,
formats them as training pairs, and runs a LoRA adaptation
on the local phi-3-mini model.

Usage:
    from server.lora_trainer import LoRATrainer
    trainer = LoRATrainer()
    result = trainer.run()  # Full pipeline: prepare -> train -> report

Requirements:
    - mlx-lm >= 0.20.0 (pip install mlx-lm)
    - Apple Silicon Mac with >= 16GB RAM
    - Base model downloaded (e.g., microsoft/phi-3-mini-4k-instruct)
"""
from __future__ import annotations

import json
import logging
import os
import platform
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger("edith.lora_trainer")

# Minimum requirements
MIN_RAM_GB = 16
DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"
DEFAULT_ITERS = 200
DEFAULT_LORA_LAYERS = 8
DEFAULT_BATCH_SIZE = 2


class LoRATrainer:
    """Local LoRA fine-tuning for Winnie's overnight sharpening.

    Pipeline:
        1. Collect corrections from autolearn.jsonl + feedback.sqlite3
        2. Format into MLX LoRA training JSONL
        3. Run LoRA training via mlx_lm.lora
        4. Save adapter to adapters/winnie-lora/
        5. Report results
    """

    def __init__(self, data_root: str = ""):
        self._data_root = Path(data_root or os.environ.get("EDITH_DATA_ROOT", "."))
        self._project_root = Path(__file__).parent.parent
        self._adapter_dir = self._project_root / "adapters" / "winnie-lora"
        self._training_dir = self._data_root / "training"
        self._model = os.environ.get("EDITH_LORA_BASE_MODEL", DEFAULT_MODEL)
        self._last_train_marker = self._adapter_dir / ".last_train"

    def is_available(self) -> bool:
        """Check if LoRA training is possible on this machine."""
        try:
            import mlx_lm  # noqa: F401
            return True
        except ImportError:
            return False

    def has_enough_ram(self) -> bool:
        """Check if machine has enough RAM for training."""
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        except ImportError:
            try:
                ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
            except Exception:
                ram_gb = 8  # Conservative fallback

        return ram_gb >= MIN_RAM_GB

    def needs_training(self) -> bool:
        """Check if there are new corrections since last training."""
        autolearn = self._project_root / "autolearn.jsonl"
        if not autolearn.exists():
            return False

        # Check if autolearn has entries since last train
        if self._last_train_marker.exists():
            last_train_time = self._last_train_marker.stat().st_mtime
            autolearn_time = autolearn.stat().st_mtime
            if autolearn_time <= last_train_time:
                return False

        # Check minimum number of corrections (stat-based, no read needed)
        try:
            size = autolearn.stat().st_size
            # Rough estimate: each entry ~200 bytes, need at least 10
            if size < 2000:
                log.info(f"[LoRA] File too small ({size} bytes) -- need more data")
                return False
        except Exception:
            return False

        return True

    def prepare_data(self) -> Path:
        """Convert autolearn.jsonl + feedback.sqlite3 -> MLX LoRA format.

        MLX LoRA expects JSONL with {"text": "<s>[INST] Q [/INST] A</s>"} format.
        """
        output_path = self._training_dir / "lora_train.jsonl"
        self._training_dir.mkdir(parents=True, exist_ok=True)

        pairs = []

        # Source 1: autolearn.jsonl (chat completions with sources)
        # §SEC: Copy file atomically to avoid data race with main.py's _autolearn_lock
        # §FIX: Support both old flat format AND new messages array format
        autolearn = self._project_root / "autolearn.jsonl"
        if autolearn.exists():
            import shutil
            tmp_copy = self._training_dir / "autolearn_snapshot.jsonl"
            try:
                shutil.copy2(str(autolearn), str(tmp_copy))
                with open(tmp_copy, "r") as fh:
                    for line in fh:
                        try:
                            entry = json.loads(line.strip())
                            q, a = "", ""
                            # New format: messages array from _save_autolearn
                            msgs = entry.get("messages", [])
                            if len(msgs) >= 3:
                                q = msgs[1].get("content", "")  # user message
                                a = msgs[2].get("content", "")  # assistant message
                            elif len(msgs) >= 2:
                                q = msgs[-2].get("content", "")
                                a = msgs[-1].get("content", "")
                            # Old format fallback: flat question/answer keys
                            if not q:
                                q = entry.get("question", entry.get("prompt", ""))
                            if not a:
                                a = entry.get("answer", entry.get("response", ""))
                            if q and a:
                                pairs.append({"text": f"<s>[INST] {q} [/INST] {a}</s>"})
                        except (json.JSONDecodeError, KeyError):
                            continue
                tmp_copy.unlink(missing_ok=True)
            except Exception as e:
                log.warning(f"[LoRA] Autolearn read failed: {e}")

        # Source 2: feedback.sqlite3 (positive feedback + corrections from UI)
        feedback_db = self._project_root / "feedback.sqlite3"
        if feedback_db.exists():
            try:
                conn = sqlite3.connect(str(feedback_db))
                cursor = conn.cursor()
                # Positive feedback — use the answer as a training example
                cursor.execute("""
                    SELECT query, answer FROM feedback_events
                    WHERE value >= 1 AND query IS NOT NULL AND answer IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT 500
                """)
                for row in cursor.fetchall():
                    q, a = row[0], row[1]
                    if q and a and len(a) > 20:
                        pairs.append({"text": f"<s>[INST] {q} [/INST] {a}</s>"})
                # Corrections — user-provided better answers (highest quality)
                cursor.execute("""
                    SELECT query, correction FROM feedback_events
                    WHERE value < 1 AND correction IS NOT NULL AND correction != ''
                    ORDER BY created_at DESC
                    LIMIT 200
                """)
                for row in cursor.fetchall():
                    q, correction = row[0], row[1]
                    if q and correction and len(correction) > 20:
                        pairs.append({"text": f"<s>[INST] {q} [/INST] {correction}</s>"})
                conn.close()
            except Exception as e:
                log.warning(f"[LoRA] Feedback DB read failed: {e}")

        # Source 3: DPO negatives — learn what NOT to say
        # §FIX T6: Search both project root AND training_data/ for negative JSONL
        dpo_negatives = []
        _neg_search_dirs = [self._project_root, self._training_dir]
        for neg_file in ["dpo_negatives.jsonl", "edith_feedback_negatives.jsonl"]:
            for _neg_dir in _neg_search_dirs:
                neg_path = _neg_dir / neg_file
                if neg_path.exists():
                    try:
                        with open(neg_path, "r") as fh:
                            for line in fh:
                                entry = json.loads(line.strip())
                                q = entry.get("question", "")
                                bad_a = entry.get("bad_answer", entry.get("answer", ""))
                                if q and bad_a:
                                    dpo_negatives.append({"question": q, "bad_answer": bad_a})
                    except Exception as e:
                        log.warning(f"[LoRA] DPO negative read from {neg_path} failed: {e}")

        # §FIX W3: Also extract DPO pairs from feedback.sqlite3 — unifies the
        # two DPO pipelines (training_tools.py SQLite + lora_trainer.py JSONL)
        if feedback_db.exists():
            try:
                import sqlite3
                conn = sqlite3.connect(str(feedback_db))
                cursor = conn.cursor()
                # Find queries that have BOTH positive and negative feedback
                cursor.execute("""
                    SELECT f1.query, f1.answer, f2.answer
                    FROM feedback_events f1
                    JOIN feedback_events f2 ON f1.query = f2.query
                    WHERE f1.value >= 1 AND f2.value < 1
                    AND f1.answer IS NOT NULL AND f2.answer IS NOT NULL
                    AND f1.answer != f2.answer
                    ORDER BY f1.created_at DESC
                    LIMIT 200
                """)
                for row in cursor.fetchall():
                    q, good_a, bad_a = row[0], row[1], row[2]
                    if q and bad_a and bad_a not in [d.get("bad_answer") for d in dpo_negatives]:
                        dpo_negatives.append({"question": q, "bad_answer": bad_a})
                conn.close()
            except Exception as e:
                log.warning(f"[LoRA] DPO SQLite extraction failed: {e}")

        if dpo_negatives:
            log.info(f"[LoRA] Found {len(dpo_negatives)} DPO negatives for contrastive learning")

        # Source 4: existing training JSONL files from Bolt
        for jsonl_file in self._training_dir.glob("*.jsonl"):
            if jsonl_file.name == "lora_train.jsonl":
                continue
            try:
                with open(jsonl_file, "r") as fh:
                    for line in fh:
                        entry = json.loads(line.strip())
                        msgs = entry.get("messages", [])
                        if len(msgs) >= 2:
                            q = msgs[-2].get("content", "")
                            a = msgs[-1].get("content", "")
                            if q and a:
                                pairs.append({"text": f"<s>[INST] {q} [/INST] {a}</s>"})
            except Exception:
                continue

        if not pairs:
            log.warning("[LoRA] No training data found")
            return Path("")

        # Write training file
        with open(output_path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")

        log.info(f"[LoRA] Prepared {len(pairs)} training pairs -> {output_path}")
        return output_path

    def train(self, iters: int = DEFAULT_ITERS, lora_layers: int = DEFAULT_LORA_LAYERS,
              batch_size: int = DEFAULT_BATCH_SIZE, max_minutes: int = 10) -> dict:
        """Run LoRA training via mlx_lm.lora subprocess.

        Args:
            iters: Number of training iterations (default 200)
            lora_layers: Number of layers to apply LoRA (default 8)
            batch_size: Training batch size (default 2)
            max_minutes: Maximum training time in minutes (default 10)

        Returns:
            dict with status, elapsed time, adapter path
        """
        training_data = self.prepare_data()
        if not training_data or not training_data.exists():
            return {"status": "no_data", "message": "No training data available"}

        self._adapter_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "-m", "mlx_lm.lora",
            "--model", self._model,
            "--data", str(training_data.parent),
            "--train",
            "--batch-size", str(batch_size),
            "--lora-layers", str(lora_layers),
            "--iters", str(iters),
            "--output", str(self._adapter_dir),
        ]

        log.info(f"[LoRA] Starting training: {iters} iters, {lora_layers} layers, "
                 f"batch={batch_size}, max={max_minutes}min")
        t0 = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True,
                timeout=max_minutes * 60,
                cwd=str(self._project_root),
            )

            elapsed = time.time() - t0

            if result.returncode == 0:
                # Mark training timestamp
                self._last_train_marker.parent.mkdir(parents=True, exist_ok=True)
                self._last_train_marker.write_text(str(time.time()))

                log.info(f"[LoRA] Training complete in {elapsed:.0f}s")
                return {
                    "status": "success",
                    "elapsed_seconds": round(elapsed, 1),
                    "adapter_path": str(self._adapter_dir),
                    "iters": iters,
                    "message": f"LoRA adapter saved to {self._adapter_dir}",
                }
            else:
                log.warning(f"[LoRA] Training failed: {result.stderr[:500]}")
                return {
                    "status": "error",
                    "elapsed_seconds": round(elapsed, 1),
                    "error": result.stderr[:500],
                }

        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            log.warning(f"[LoRA] Training timed out after {max_minutes} minutes")
            return {
                "status": "timeout",
                "elapsed_seconds": round(elapsed, 1),
                "message": f"Training exceeded {max_minutes} minute limit",
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def adapter_exists(self) -> bool:
        """Check if a trained LoRA adapter exists."""
        return (self._adapter_dir / "adapter_config.json").exists() or \
               (self._adapter_dir / "adapters.safetensors").exists()

    def get_adapter_info(self) -> dict:
        """Get information about the current adapter."""
        if not self.adapter_exists():
            return {"exists": False}

        info = {"exists": True, "path": str(self._adapter_dir)}

        if self._last_train_marker.exists():
            ts = float(self._last_train_marker.read_text().strip())
            info["last_trained"] = ts
            info["last_trained_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))

        # Count adapter files
        adapter_files = list(self._adapter_dir.glob("*"))
        info["files"] = len(adapter_files)
        info["size_mb"] = round(sum(f.stat().st_size for f in adapter_files if f.is_file()) / (1024 ** 2), 1)

        return info

    def run(self) -> dict:
        """Full pipeline: check -> prepare -> train -> report."""
        if not self.is_available():
            return {"status": "unavailable", "message": "mlx-lm not installed"}

        if not self.has_enough_ram():
            return {"status": "insufficient_ram",
                    "message": f"Need >= {MIN_RAM_GB}GB RAM for LoRA training"}

        if not self.needs_training():
            return {"status": "up_to_date",
                    "message": "No new corrections since last training",
                    "adapter": self.get_adapter_info()}

        return self.train()
