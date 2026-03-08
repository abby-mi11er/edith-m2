"""
MLX Local Inference — On-Device Generation
============================================
Runs a small language model locally on Apple Silicon via mlx-lm
for quick queries that don't need Gemini's full reasoning.

On M4: ~60 tok/s (good for "quick" depth answers)
On M2: ~25 tok/s (acceptable for short factual answers)

Only used when:
- depth == "quick"
- Profile mode allows local inference
- Model is downloaded

Falls back to Gemini API if any condition fails.
"""

import logging
import os
import time
from typing import Generator, Optional

log = logging.getLogger("edith.mlx_infer")

_mlx_lm_available = False
_model = None
_tokenizer = None
_model_name: str = ""
_loading = False

# Small models that fit in M2's 8GB and run fast on M4
# Profile-driven: M4 gets 1.5B (better reasoning), M2 gets 0.5B (lighter)
_DEFAULT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

# §M2-1: KV cache quantization bits — 4-bit on M2 saves ~50% memory
_KV_CACHE_BITS = int(os.getenv("KV_CACHE_BITS", "8"))
_MAX_MEMORY_GB = int(os.getenv("MAX_MEMORY_GB", "0"))
_METAL_CACHE_APPLIED = False

def _get_profile_model() -> str:
    """Get the local model from the compute profile."""
    try:
        from server.backend_logic import get_compute_profile
        return get_compute_profile().get("local_model", _DEFAULT_MODEL)
    except Exception:
        return _DEFAULT_MODEL

# Fallback if Qwen isn't available
_FALLBACK_MODELS = [
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit",  # ~0.5GB — very fast
    "mlx-community/SmolLM2-360M-Instruct-4bit",   # ~0.2GB — minimal
]


def _check_mlx_lm():
    """Check if mlx-lm is available."""
    global _mlx_lm_available
    try:
        import mlx_lm
        _mlx_lm_available = True
        return True
    except ImportError:
        log.info("§LOCAL: mlx-lm not installed — local inference disabled")
        _mlx_lm_available = False
        return False


def _apply_metal_cache_limit():
    """§M2-1: Cap MLX GPU cache to prevent OS/Electron memory starvation.

    On 8GB M2 Air, sets limit to 5.5GB so macOS + Electron keep ~2.5GB.
    On M4 Pro (16GB+), no cap is applied.
    """
    global _METAL_CACHE_APPLIED
    if _METAL_CACHE_APPLIED:
        return
    _METAL_CACHE_APPLIED = True

    ram_gb = _MAX_MEMORY_GB
    if ram_gb == 0:
        # Auto-detect
        try:
            import subprocess
            mem = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=3,
            )
            ram_gb = round(int(mem.stdout.strip()) / (1024**3))
        except Exception:
            ram_gb = 16  # assume high-end if detection fails

    if ram_gb <= 8:
        try:
            import mlx.core as mx
            # Cap at 5.5GB — leave ~2.5GB for OS + Electron + Python
            limit = int(5.5 * 1024 * 1024 * 1024)
            mx.metal.set_cache_limit(limit)
            log.info(f"§M2-1: Metal cache capped at 5.5GB (RAM={ram_gb}GB)")
        except Exception as e:
            log.warning(f"§M2-1: Could not set metal cache limit: {e}")
    else:
        log.info(f"§M2-1: RAM={ram_gb}GB — no metal cache cap needed")


def _clear_metal_cache():
    """§M2-1: Aggressively clear MLX cache after generation on memory-constrained machines."""
    if _MAX_MEMORY_GB and _MAX_MEMORY_GB > 8:
        return  # Only clear on M2-class machines
    try:
        import mlx.core as mx
        mx.metal.clear_cache()
    except Exception:
        pass


def is_available() -> bool:
    """Check if local inference is available (mlx-lm installed + model loaded)."""
    if not _mlx_lm_available and not _check_mlx_lm():
        return False
    return True


def load_model(model_name: str = "") -> bool:
    """Pre-load the model. Call during warmup to avoid first-query latency.

    Returns True if model is loaded successfully.
    """
    global _model, _tokenizer, _model_name, _loading

    if _loading:
        return False
    if _model is not None and (_model_name == (model_name or _get_profile_model())):
        return True
    if not is_available():
        return False

    _loading = True
    model_name = model_name or _get_profile_model()
    models_to_try = [model_name] + [m for m in _FALLBACK_MODELS if m != model_name]

    # §M2-1: Apply metal cache limit before loading
    _apply_metal_cache_limit()

    # §M2-1: KV cache quantization kwargs
    load_kwargs = {}
    if _KV_CACHE_BITS < 8:
        load_kwargs["kv_bits"] = _KV_CACHE_BITS
        log.info(f"§M2-1: KV cache quantization enabled at {_KV_CACHE_BITS}-bit")

    for name in models_to_try:
        try:
            import mlx_lm
            log.info(f"§LOCAL: Loading model: {name}")
            t0 = time.time()
            _model, _tokenizer = mlx_lm.load(name, **load_kwargs)
            _model_name = name
            elapsed = time.time() - t0
            log.info(f"§LOCAL: Model loaded in {elapsed:.1f}s: {name}")
            _loading = False
            return True
        except Exception as e:
            log.warning(f"§LOCAL: Failed to load {name}: {e}")
            continue

    log.warning("§LOCAL: No local model available — will use API")
    _loading = False
    return False


def generate(
    prompt: str,
    system_instruction: str = "",
    max_tokens: int = 512,
    temperature: float = 0.1,
) -> Optional[str]:
    """Generate text locally using MLX.

    Args:
        prompt: The user prompt
        system_instruction: System-level instruction
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated text, or None if local inference unavailable
    """
    if _model is None:
        if not load_model():
            return None

    try:
        import mlx_lm

        # Build chat messages
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        # Apply chat template
        chat_prompt = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        t0 = time.time()
        result = mlx_lm.generate(
            _model,
            _tokenizer,
            prompt=chat_prompt,
            max_tokens=max_tokens,
            temp=temperature,
            verbose=False,
        )
        # mlx_lm.generate returns GenerationResponse with .text field
        if hasattr(result, 'text'):
            response_text = result.text or ""
        elif isinstance(result, str):
            response_text = result
        else:
            response_text = str(result)
        elapsed = time.time() - t0

        # Estimate throughput
        token_count = len(_tokenizer.encode(response_text))
        tps = token_count / elapsed if elapsed > 0 else 0
        log.info(f"§LOCAL: Generated {token_count} tokens in {elapsed:.1f}s "
                 f"({tps:.0f} tok/s) via {_model_name}")

        return response_text.strip()
    except Exception as e:
        log.error(f"§LOCAL: Generation failed: {e}")
        return None
    finally:
        _clear_metal_cache()


def generate_stream(
    prompt: str,
    system_instruction: str = "",
    max_tokens: int = 512,
    temperature: float = 0.1,
) -> Generator[str, None, None]:
    """Stream tokens locally using MLX.

    Yields individual tokens as they are generated.
    Falls back silently (yields nothing) if unavailable.
    """
    if _model is None:
        if not load_model():
            return

    try:
        import mlx_lm

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        chat_prompt = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        t0 = time.time()
        token_count = 0

        for response in mlx_lm.stream_generate(
            _model,
            _tokenizer,
            prompt=chat_prompt,
            max_tokens=max_tokens,
            temp=temperature,
        ):
            # mlx_lm.stream_generate yields GenerationResponse objects
            if hasattr(response, 'text'):
                text = response.text or ""
            elif isinstance(response, dict):
                text = response.get("text", "")
            else:
                text = str(response)
            if text:
                token_count += 1
                yield text

        elapsed = time.time() - t0
        tps = token_count / elapsed if elapsed > 0 else 0
        log.info(f"§LOCAL: Streamed {token_count} tokens in {elapsed:.1f}s "
                 f"({tps:.0f} tok/s) via {_model_name}")
        _clear_metal_cache()
    except Exception as e:
        log.error(f"§LOCAL: Streaming failed: {e}")
        _clear_metal_cache()
        return


def get_model_info() -> dict:
    """Return info about the loaded model."""
    return {
        "loaded": _model is not None,
        "model": _model_name,
        "mlx_available": _mlx_lm_available,
    }
