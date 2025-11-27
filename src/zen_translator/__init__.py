"""
Zen Translator - Real-time multimodal translation with lip sync and voice cloning.

Built on:
- Qwen3-Omni: Real-time speech understanding and translation
- CosyVoice 2.0: Ultra-low latency voice cloning (150ms)
- Wav2Lip: Accurate lip synchronization

Features:
- 18 input languages, 10 output languages
- News anchor voice finetuning for accurate translation
- Sub-second end-to-end latency
- WebRTC streaming support
"""

__version__ = "0.1.0"
__author__ = "Hanzo AI / Zen LM"

from .config import TranslatorConfig
from .pipeline import TranslationPipeline

__all__ = [
    "TranslationPipeline",
    "TranslatorConfig",
    "__version__",
]
