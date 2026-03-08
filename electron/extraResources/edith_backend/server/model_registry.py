"""
Model Registry — Auto-detect local AI models
===============================================
Scans the local system for available AI models from:
  - Ollama (local LLMs)
  - LM Studio (local inference)
  - Hugging Face cache (~/.cache/huggingface/)
  - MLX models (~/.cache/mlx/)
  - OpenAI API (remote, if key set)
  - Gemini API (remote, if key set)

Provides a unified model list for the SettingsPanel dropdown.

Usage:
    from server.model_registry import ModelRegistry
    registry = ModelRegistry()
    models = registry.discover()
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

log = logging.getLogger("edith.model_registry")


class ModelInfo:
    """Metadata for a discovered model."""

    def __init__(self, name: str, provider: str, model_type: str = "chat",
                 path: str = "", size_gb: float = 0, quantization: str = "",
                 available: bool = True):
        self.name = name
        self.provider = provider
        self.model_type = model_type  # chat, embed, vision
        self.path = path
        self.size_gb = size_gb
        self.quantization = quantization
        self.available = available

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "provider": self.provider,
            "type": self.model_type,
            "path": self.path,
            "size_gb": self.size_gb,
            "quantization": self.quantization,
            "available": self.available,
        }


class ModelRegistry:
    """Discover and manage available AI models."""

    def __init__(self):
        self._models: list[ModelInfo] = []

    def discover(self) -> list[dict]:
        """Run full model discovery across all providers."""
        self._models = []
        self._scan_ollama()
        self._scan_lm_studio()
        self._scan_huggingface_cache()
        self._scan_mlx_cache()
        self._scan_api_models()
        return [m.to_dict() for m in self._models]

    def _scan_ollama(self):
        """Check for Ollama models (local LLM runner)."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")[1:]  # Skip header
                for line in lines:
                    parts = line.split()
                    if parts:
                        name = parts[0]
                        size_str = parts[2] if len(parts) > 2 else ""
                        size_gb = 0
                        if "GB" in size_str:
                            try:
                                size_gb = float(size_str.replace("GB", ""))
                            except ValueError:
                                pass
                        self._models.append(ModelInfo(
                            name=name, provider="ollama",
                            model_type="chat", size_gb=size_gb,
                        ))
                log.info(f"[ModelRegistry] Ollama: {len(lines)} model(s) found")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass  # Ollama not installed

    def _scan_lm_studio(self):
        """Check for LM Studio downloaded models."""
        lm_dir = Path.home() / ".cache" / "lm-studio" / "models"
        if not lm_dir.exists():
            lm_dir = Path.home() / ".lmstudio" / "models"
        if not lm_dir.exists():
            return

        for model_dir in lm_dir.rglob("*.gguf"):
            size_gb = round(model_dir.stat().st_size / (1024 ** 3), 1)
            self._models.append(ModelInfo(
                name=model_dir.stem,
                provider="lm_studio",
                model_type="chat",
                path=str(model_dir),
                size_gb=size_gb,
                quantization=self._detect_quant(model_dir.name),
            ))
        log.info(f"[ModelRegistry] LM Studio: {sum(1 for m in self._models if m.provider == 'lm_studio')} model(s)")

    def _scan_huggingface_cache(self):
        """Check Hugging Face local cache for downloaded models."""
        hf_dir = Path.home() / ".cache" / "huggingface" / "hub"
        if not hf_dir.exists():
            return

        for model_dir in hf_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue
            # Extract model name from directory (models--org--name format)
            name = model_dir.name.replace("models--", "").replace("--", "/")
            # Check for GGUF files
            gguf_files = list(model_dir.rglob("*.gguf"))
            safetensor_files = list(model_dir.rglob("*.safetensors"))

            if gguf_files or safetensor_files:
                total_size = sum(f.stat().st_size for f in (gguf_files + safetensor_files))
                self._models.append(ModelInfo(
                    name=name,
                    provider="huggingface",
                    model_type="chat",
                    path=str(model_dir),
                    size_gb=round(total_size / (1024 ** 3), 1),
                ))
        log.info(f"[ModelRegistry] HuggingFace: {sum(1 for m in self._models if m.provider == 'huggingface')} model(s)")

    def _scan_mlx_cache(self):
        """Check for MLX-format models (Apple Silicon optimized)."""
        mlx_dir = Path.home() / ".cache" / "huggingface" / "hub"
        if not mlx_dir.exists():
            return

        for model_dir in mlx_dir.iterdir():
            if not model_dir.is_dir():
                continue
            # Look for MLX weight files
            mlx_files = list(model_dir.rglob("*.npz")) + list(model_dir.rglob("weights.*.safetensors"))
            config = model_dir / "snapshots"
            if mlx_files and "mlx" in model_dir.name.lower():
                name = model_dir.name.replace("models--", "").replace("--", "/")
                total_size = sum(f.stat().st_size for f in mlx_files)
                self._models.append(ModelInfo(
                    name=name,
                    provider="mlx",
                    model_type="chat",
                    path=str(model_dir),
                    size_gb=round(total_size / (1024 ** 3), 1),
                ))

    def _scan_api_models(self):
        """Register remote API models (OpenAI, Gemini) if keys are set."""
        if os.environ.get("OPENAI_API_KEY"):
            for model in ["gpt-4o", "gpt-4o-mini", "text-embedding-3-small"]:
                mtype = "embed" if "embedding" in model else "chat"
                self._models.append(ModelInfo(
                    name=model, provider="openai", model_type=mtype,
                ))
            # Check for fine-tuned model
            ft = os.environ.get("WINNIE_OPENAI_MODEL", "")
            if ft:
                self._models.append(ModelInfo(
                    name=ft, provider="openai_finetune", model_type="chat",
                ))

        if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
            for model in ["gemini-2.5-flash", "models/embedding-001"]:
                mtype = "embed" if "embedding" in model else "chat"
                self._models.append(ModelInfo(
                    name=model, provider="gemini", model_type=mtype,
                ))

    @staticmethod
    def _detect_quant(filename: str) -> str:
        """Detect quantization level from filename."""
        lower = filename.lower()
        for q in ["q2_k", "q3_k", "q4_0", "q4_k_m", "q5_0", "q5_k_m",
                   "q6_k", "q8_0", "f16", "f32"]:
            if q in lower:
                return q.upper()
        return ""

    def get_by_provider(self, provider: str) -> list[dict]:
        return [m.to_dict() for m in self._models if m.provider == provider]

    def get_chat_models(self) -> list[dict]:
        return [m.to_dict() for m in self._models if m.model_type == "chat"]

    def get_embed_models(self) -> list[dict]:
        return [m.to_dict() for m in self._models if m.model_type == "embed"]
