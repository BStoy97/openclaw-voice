#!/usr/bin/env python3
"""
Download Whisper (STT) and Piper (TTS) models for offline use.

Usage:
    python scripts/download_models.py [model_name]
    python scripts/download_models.py piper [voice_name]

Whisper models:
    tiny    - 39M params, ~1GB VRAM (fastest)
    base    - 74M params, ~1GB VRAM
    small   - 244M params, ~2GB VRAM
    medium  - 769M params, ~5GB VRAM
    large-v3-turbo - 809M params, ~6GB VRAM (best quality/speed)

Piper voices (local TTS, default backend):
    en_US-amy-medium    - default female en_US voice
    en_US-lessac-medium - fallback if amy is unavailable

Smart-turn (semantic end-of-turn detection):
    python scripts/download_models.py smart-turn
"""

import sys
import os
import urllib.request
from pathlib import Path

PIPER_VOICES_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
PIPER_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "piper"
DEFAULT_PIPER_VOICE = "en_US-amy-medium"
FALLBACK_PIPER_VOICE = "en_US-lessac-medium"

# pipecat-ai smart-turn-v3 (BSD-2-Clause): semantic end-of-turn classifier.
# int8-quantized CPU checkpoint (~8 MB); verified filename on the HF repo.
SMART_TURN_URL = (
    "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/"
    "smart-turn-v3.2-cpu.onnx"
)
SMART_TURN_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "smart-turn"


def download_model(model_name: str = "base"):
    """Download a Whisper model."""
    print(f"Downloading Whisper model: {model_name}")
    print("This may take a few minutes...")
    
    try:
        from faster_whisper import WhisperModel
        
        # This will download the model if not cached
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        
        print(f"✅ Model '{model_name}' downloaded successfully!")
        print(f"   Cached at: ~/.cache/huggingface/")
        
        # Test the model
        import numpy as np
        audio = np.zeros(16000, dtype=np.float32)
        segments, info = model.transcribe(audio)
        list(segments)  # Consume generator
        
        print(f"✅ Model tested successfully!")
        
    except ImportError:
        print("❌ faster-whisper not installed. Run:")
        print("   pip install faster-whisper")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def _piper_voice_url(voice_name: str, ext: str) -> str:
    """en_US-amy-medium -> https://.../en/en_US/amy/medium/en_US-amy-medium.<ext>"""
    lang_country, speaker, quality = voice_name.split("-")
    lang = lang_country.split("_")[0]
    return f"{PIPER_VOICES_BASE}/{lang}/{lang_country}/{speaker}/{quality}/{voice_name}.{ext}"


def download_piper_voice(voice_name: str = DEFAULT_PIPER_VOICE) -> bool:
    """Download a Piper voice (.onnx + .onnx.json) into models/piper/.

    Falls back to en_US-lessac-medium if the requested voice fails to download.
    """
    candidates = [voice_name]
    if voice_name != FALLBACK_PIPER_VOICE:
        candidates.append(FALLBACK_PIPER_VOICE)

    for candidate in candidates:
        model_path = PIPER_MODEL_DIR / f"{candidate}.onnx"
        config_path = PIPER_MODEL_DIR / f"{candidate}.onnx.json"

        if model_path.exists() and config_path.exists():
            print(f"✅ Piper voice '{candidate}' already present at {PIPER_MODEL_DIR}")
            return True

        try:
            PIPER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            print(f"Downloading Piper voice: {candidate}")
            urllib.request.urlretrieve(_piper_voice_url(candidate, "onnx"), model_path)
            urllib.request.urlretrieve(_piper_voice_url(candidate, "onnx.json"), config_path)
            print(f"✅ Downloaded '{candidate}' to {PIPER_MODEL_DIR}")
            print("   Set OPENCLAW_TTS_VOICE to use a non-default voice.")
            return True
        except Exception as e:
            print(f"❌ Failed to download '{candidate}': {e}")
            model_path.unlink(missing_ok=True)
            config_path.unlink(missing_ok=True)

    print("❌ All Piper voice download attempts failed.")
    return False


def download_smart_turn() -> bool:
    """Download the smart-turn-v3 ONNX model into models/smart-turn/."""
    filename = SMART_TURN_URL.rsplit("/", 1)[-1]
    dest = SMART_TURN_MODEL_DIR / filename

    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"✅ Smart-turn model already present: {dest}")
        return True

    try:
        SMART_TURN_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading smart-turn model: {filename}")
        urllib.request.urlretrieve(SMART_TURN_URL, dest)
        if dest.stat().st_size < 1_000_000:
            raise IOError(f"Downloaded file suspiciously small ({dest.stat().st_size} bytes)")
        print(f"✅ Downloaded smart-turn model to {dest}")
        return True
    except Exception as e:
        print(f"❌ Failed to download smart-turn model: {e}")
        dest.unlink(missing_ok=True)
        return False


def list_models():
    """List available models."""
    models = {
        "tiny": "39M params, ~1GB VRAM, fastest",
        "base": "74M params, ~1GB VRAM, good balance",
        "small": "244M params, ~2GB VRAM",
        "medium": "769M params, ~5GB VRAM",
        "large-v3": "1.5B params, ~10GB VRAM, best quality",
        "large-v3-turbo": "809M params, ~6GB VRAM, best quality/speed ratio",
    }
    
    print("Available Whisper models:")
    print()
    for name, desc in models.items():
        print(f"  {name:20} - {desc}")
    print()
    print("Recommended: large-v3-turbo (GPU) or base (CPU)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        list_models()
        print()
        model = input("Enter model name to download (or 'q' to quit): ").strip()
        if model.lower() == 'q':
            sys.exit(0)
    else:
        model = sys.argv[1]

        if model in ["-h", "--help", "help"]:
            list_models()
            sys.exit(0)

        if model == "piper":
            voice = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PIPER_VOICE
            ok = download_piper_voice(voice)
            sys.exit(0 if ok else 1)

        if model in ("smart-turn", "smart_turn"):
            ok = download_smart_turn()
            sys.exit(0 if ok else 1)

    download_model(model)
