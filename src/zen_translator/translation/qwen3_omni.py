"""
Qwen3-Omni translation module.

Real-time multimodal translation using Qwen3-Omni-30B-A3B.
Supports audio, video, and text input with real-time speech output.
"""

import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from transformers import AutoProcessor

from ..config import TranslatorConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Lazy import for Qwen3-Omni model (may not be available in all transformers versions)
Qwen3OmniForConditionalGeneration = None


class Qwen3OmniTranslator:
    """Real-time translation using Qwen3-Omni."""

    def __init__(self, config: TranslatorConfig):
        self.config = config
        self.model = None
        self.processor = None
        self._loaded = False

    def load(self) -> None:
        """Load the Qwen3-Omni model."""
        if self._loaded:
            return

        logger.info(f"Loading Qwen3-Omni from {self.config.qwen3_omni_model}")

        # Lazy import the model class
        global Qwen3OmniForConditionalGeneration
        if Qwen3OmniForConditionalGeneration is None:
            try:
                from transformers import Qwen3OmniForConditionalGeneration as _Qwen3Omni

                Qwen3OmniForConditionalGeneration = _Qwen3Omni
            except ImportError:
                # Fall back to AutoModelForCausalLM with trust_remote_code
                from transformers import AutoModelForCausalLM

                Qwen3OmniForConditionalGeneration = AutoModelForCausalLM
                logger.warning(
                    "Qwen3OmniForConditionalGeneration not available, "
                    "using AutoModelForCausalLM with trust_remote_code"
                )

        # Determine torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[self.config.dtype]

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.config.qwen3_omni_model,
            cache_dir=self.config.model_cache_dir,
            trust_remote_code=True,
        )

        # Load model with optimizations
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": "auto",
            "cache_dir": self.config.model_cache_dir,
            "trust_remote_code": True,
        }

        if self.config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = Qwen3OmniForConditionalGeneration.from_pretrained(
            self.config.qwen3_omni_model,
            **model_kwargs,
        )

        if self.config.compile_model:
            logger.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model, mode="reduce-overhead")

        self._loaded = True
        logger.info("Qwen3-Omni loaded successfully")

    def unload(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._loaded = False
        torch.cuda.empty_cache()

    async def translate_audio(
        self,
        audio: np.ndarray | Path | str,
        source_lang: str | None = None,
        target_lang: str | None = None,
        return_audio: bool = True,
    ) -> dict:
        """
        Translate audio input to target language.

        Args:
            audio: Audio as numpy array, file path, or URL
            source_lang: Source language (auto-detect if None)
            target_lang: Target language
            return_audio: Whether to return synthesized audio

        Returns:
            dict with keys: text, audio (optional), source_lang, target_lang
        """
        if not self._loaded:
            self.load()

        source_lang = source_lang or self.config.source_language
        target_lang = target_lang or self.config.target_language

        # Build translation prompt
        system_prompt = self._build_translation_prompt(source_lang, target_lang)

        # Process audio input
        if isinstance(audio, (str, Path)):
            audio_input = str(audio)
        else:
            audio_input = audio

        # Create conversation format
        conversation = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_input},
                    {"type": "text", "text": f"Translate this speech to {target_lang}."},
                ],
            },
        ]

        # Process with Qwen3-Omni processor
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate translation with optional audio output
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                return_audio=return_audio,
                audio_output_config={
                    "sample_rate": 24000,
                    "speaker_id": 0,  # Will be overridden by voice cloning
                },
            )

        # Decode outputs
        text_output = self.processor.decode(
            outputs.sequences[0],
            skip_special_tokens=True,
        )

        result = {
            "text": text_output,
            "source_lang": source_lang,
            "target_lang": target_lang,
        }

        if return_audio and hasattr(outputs, "audio"):
            result["audio"] = outputs.audio[0].cpu().numpy()
            result["sample_rate"] = 24000

        return result

    async def translate_video(
        self,
        video: Path | str,
        source_lang: str | None = None,
        target_lang: str | None = None,
    ) -> dict:
        """
        Translate video with lip-reading enhancement.

        Uses visual context (lip movements, gestures, on-screen text)
        to improve translation accuracy in noisy environments.
        """
        if not self._loaded:
            self.load()

        source_lang = source_lang or self.config.source_language
        target_lang = target_lang or self.config.target_language

        # Build enhanced prompt for video
        system_prompt = self._build_video_translation_prompt(source_lang, target_lang)

        conversation = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video)},
                    {
                        "type": "text",
                        "text": (
                            f"Translate the speech in this video to {target_lang}. "
                            "Use visual context (lip movements, gestures, on-screen text) "
                            "to improve accuracy."
                        ),
                    },
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
                return_audio=True,
            )

        text_output = self.processor.decode(
            outputs.sequences[0],
            skip_special_tokens=True,
        )

        return {
            "text": text_output,
            "audio": outputs.audio[0].cpu().numpy() if hasattr(outputs, "audio") else None,
            "sample_rate": 24000,
            "source_lang": source_lang,
            "target_lang": target_lang,
        }

    async def stream_translate(
        self,
        audio_stream: AsyncIterator[np.ndarray],
        source_lang: str | None = None,
        target_lang: str | None = None,
    ) -> AsyncIterator[dict]:
        """
        Stream translation for real-time applications.

        Yields translation chunks as they become available.
        """
        if not self._loaded:
            self.load()

        source_lang = source_lang or self.config.source_language
        target_lang = target_lang or self.config.target_language

        # Buffer for accumulating audio chunks
        buffer = []
        chunk_duration_ms = self.config.streaming_chunk_ms
        sample_rate = 16000  # Expected input sample rate
        chunk_samples = int(sample_rate * chunk_duration_ms / 1000)

        async for audio_chunk in audio_stream:
            buffer.append(audio_chunk)
            total_samples = sum(len(c) for c in buffer)

            # Process when we have enough audio
            if total_samples >= chunk_samples:
                combined = np.concatenate(buffer)
                buffer = []

                # Translate chunk
                result = await self.translate_audio(
                    combined,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    return_audio=True,
                )

                yield result

    def _build_translation_prompt(self, source_lang: str, target_lang: str) -> str:
        """Build system prompt for translation."""
        return f"""You are Zen Translator, a real-time multilingual translation system.

Your task is to translate speech from {source_lang if source_lang != "auto" else "the detected language"} to {target_lang}.

Guidelines:
1. Preserve the speaker's tone, emotion, and intent
2. Maintain natural speech patterns in the target language
3. Handle idiomatic expressions appropriately
4. Preserve proper nouns and technical terms when appropriate
5. Output natural, fluent {target_lang} speech

For news anchor translations:
- Maintain professional broadcast tone
- Preserve urgency and emphasis patterns
- Handle specialized news vocabulary accurately
- Keep translations concise and clear"""

    def _build_video_translation_prompt(self, source_lang: str, target_lang: str) -> str:
        """Build system prompt for video translation with visual context."""
        return f"""You are Zen Translator, a real-time multimodal translation system.

Your task is to translate the video content from {source_lang if source_lang != "auto" else "the detected language"} to {target_lang}.

You have access to both audio and visual information:
- Speech audio for primary content
- Lip movements for disambiguation in noisy audio
- Gestures and body language for context
- On-screen text (captions, graphics) for verification
- Visual scene context for improved understanding

Guidelines:
1. Use visual cues to resolve ambiguous audio
2. Reference on-screen text to verify proper nouns and numbers
3. Consider speaker's expressions for emotional context
4. Handle multiple speakers by tracking visual positions
5. Maintain synchronization awareness for lip-sync downstream

Output the translation maintaining natural {target_lang} speech patterns."""
