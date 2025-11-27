"""
Main translation pipeline orchestrating all components.

Combines Qwen3-Omni, CosyVoice, and Wav2Lip for end-to-end
real-time translation with voice cloning and lip sync.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np

from .config import TranslatorConfig
from .lip_sync import Wav2LipSync
from .translation import Qwen3OmniTranslator
from .voice_clone import CosyVoiceCloner, NewsAnchorVoiceBank

logger = logging.getLogger(__name__)


class TranslationPipeline:
    """
    End-to-end translation pipeline with voice cloning and lip sync.

    Pipeline stages:
    1. Audio/Video input → Qwen3-Omni (translation + understanding)
    2. Translated text → CosyVoice (voice synthesis in cloned voice)
    3. Cloned audio + Video → Wav2Lip (lip synchronization)

    Total latency target: <1 second end-to-end
    """

    def __init__(self, config: TranslatorConfig | None = None):
        self.config = config or TranslatorConfig()

        # Initialize components
        self.translator = Qwen3OmniTranslator(self.config)
        self.voice_cloner = CosyVoiceCloner(self.config)
        self.lip_sync = Wav2LipSync(self.config)

        # News anchor voice bank
        self.anchor_voices = NewsAnchorVoiceBank(
            self.voice_cloner,
            self.config.model_cache_dir / "voices" / "anchors",
        )

        self._loaded = False

    async def load(self) -> None:
        """Load all models."""
        if self._loaded:
            return

        logger.info("Loading translation pipeline components...")

        # Load models in parallel where possible
        await asyncio.gather(
            asyncio.to_thread(self.translator.load),
            asyncio.to_thread(self.voice_cloner.load),
            asyncio.to_thread(self.lip_sync.load)
            if self.config.enable_lip_sync
            else asyncio.sleep(0),
        )

        self._loaded = True
        logger.info("Translation pipeline loaded successfully")

    async def unload(self) -> None:
        """Unload all models to free memory."""
        self.translator.unload()
        self.voice_cloner.unload()
        self.lip_sync.unload()
        self._loaded = False

    async def translate_audio(
        self,
        audio: np.ndarray | Path | str,
        source_lang: str | None = None,
        target_lang: str | None = None,
        speaker_id: str | None = None,
    ) -> dict:
        """
        Translate audio and optionally clone voice.

        Args:
            audio: Input audio
            source_lang: Source language (auto-detect if None)
            target_lang: Target language
            speaker_id: Registered speaker for voice cloning

        Returns:
            dict with text, audio, and metadata
        """
        if not self._loaded:
            await self.load()

        # Step 1: Translate with Qwen3-Omni
        translation = await self.translator.translate_audio(
            audio,
            source_lang=source_lang,
            target_lang=target_lang,
            return_audio=speaker_id is None,  # Use Qwen3-Omni TTS if no cloning
        )

        result = {
            "text": translation["text"],
            "source_lang": translation["source_lang"],
            "target_lang": translation["target_lang"],
        }

        # Step 2: Voice cloning (if speaker registered)
        if speaker_id and speaker_id in self.voice_cloner.speaker_embeddings:
            cloned = await self.voice_cloner.clone_voice(
                text=translation["text"],
                speaker_id=speaker_id,
                language=target_lang or self.config.target_language,
            )
            result["audio"] = cloned["audio"]
            result["sample_rate"] = cloned["sample_rate"]
            result["speaker_id"] = speaker_id
        elif "audio" in translation:
            result["audio"] = translation["audio"]
            result["sample_rate"] = translation.get("sample_rate", 24000)

        return result

    async def translate_video(
        self,
        video: Path | str,
        source_lang: str | None = None,
        target_lang: str | None = None,
        speaker_id: str | None = None,
        output_path: Path | None = None,
    ) -> dict:
        """
        Translate video with lip sync.

        Full pipeline:
        1. Extract audio/video analysis with Qwen3-Omni
        2. Translate speech to target language
        3. Clone voice with CosyVoice
        4. Synchronize lips with Wav2Lip

        Args:
            video: Input video path
            source_lang: Source language
            target_lang: Target language
            speaker_id: Speaker for voice cloning (uses original voice profile if None)
            output_path: Output video path

        Returns:
            dict with output path and translation details
        """
        if not self._loaded:
            await self.load()

        video_path = Path(video)

        # Step 1: Extract and analyze video with Qwen3-Omni
        logger.info("Analyzing video with Qwen3-Omni...")
        translation = await self.translator.translate_video(
            video_path,
            source_lang=source_lang,
            target_lang=target_lang,
        )

        result = {
            "text": translation["text"],
            "source_lang": translation["source_lang"],
            "target_lang": translation["target_lang"],
        }

        # Step 2: Register speaker from original video if needed
        if speaker_id is None:
            # Extract voice from original video for cloning
            speaker_id = f"video_{video_path.stem}"
            await self._register_speaker_from_video(video_path, speaker_id)

        # Step 3: Clone voice with translated text
        logger.info(f"Cloning voice with speaker: {speaker_id}")
        cloned = await self.voice_cloner.clone_voice(
            text=translation["text"],
            speaker_id=speaker_id,
            language=target_lang or self.config.target_language,
        )

        result["audio"] = cloned["audio"]
        result["sample_rate"] = cloned["sample_rate"]
        result["speaker_id"] = speaker_id

        # Step 4: Lip synchronization
        if self.config.enable_lip_sync:
            logger.info("Synchronizing lips with Wav2Lip...")

            if output_path is None:
                output_path = video_path.parent / f"{video_path.stem}_translated.mp4"

            lip_result = await self.lip_sync.sync_video(
                video=video_path,
                audio=cloned["audio"],
                output_path=output_path,
                audio_sample_rate=cloned["sample_rate"],
            )

            result["output_path"] = lip_result["output_path"]
            result["frame_count"] = lip_result.get("frame_count")

        return result

    async def stream_translate(
        self,
        audio_stream: AsyncIterator[np.ndarray],
        source_lang: str | None = None,
        target_lang: str | None = None,
        speaker_id: str | None = None,
    ) -> AsyncIterator[dict]:
        """
        Stream translation for real-time applications.

        Yields translation chunks as they become available.
        Target first-packet latency: <500ms
        """
        if not self._loaded:
            await self.load()

        # Create streaming translation
        async for translation_chunk in self.translator.stream_translate(
            audio_stream,
            source_lang=source_lang,
            target_lang=target_lang,
        ):
            # Clone voice for this chunk
            if speaker_id and speaker_id in self.voice_cloner.speaker_embeddings:
                async for voice_chunk in self.voice_cloner.stream_clone(
                    self._text_chunks(translation_chunk["text"]),
                    speaker_id=speaker_id,
                    language=target_lang or self.config.target_language,
                ):
                    yield {
                        "text": voice_chunk["text"],
                        "audio": voice_chunk["audio"],
                        "sample_rate": voice_chunk["sample_rate"],
                        "source_lang": translation_chunk.get("source_lang"),
                        "target_lang": translation_chunk.get("target_lang"),
                    }
            else:
                yield translation_chunk

    async def _text_chunks(self, text: str) -> AsyncIterator[str]:
        """Convert text to async iterator of chunks."""
        yield text

    async def _register_speaker_from_video(
        self,
        video_path: Path,
        speaker_id: str,
    ) -> None:
        """Extract and register speaker voice from video."""
        import subprocess
        import tempfile

        # Extract audio from video
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_audio = f.name

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vn",  # No video
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                temp_audio,
            ],
            check=True,
            capture_output=True,
        )

        # Register speaker
        await self.voice_cloner.register_speaker(
            speaker_id=speaker_id,
            reference_audio=temp_audio,
            sample_rate=16000,
        )

        # Cleanup
        Path(temp_audio).unlink()

    async def register_speaker(
        self,
        speaker_id: str,
        reference_audio: np.ndarray | Path | str,
        sample_rate: int = 16000,
    ) -> dict:
        """Register a speaker for voice cloning."""
        return await self.voice_cloner.register_speaker(
            speaker_id=speaker_id,
            reference_audio=reference_audio,
            sample_rate=sample_rate,
        )

    async def load_news_anchors(self) -> dict[str, bool]:
        """Load all pre-registered news anchor voices."""
        return await self.anchor_voices.load_all_voices()

    def get_supported_languages(self) -> dict:
        """Get supported input and output languages."""
        return {
            "input": self.config.supported_input_languages,
            "output": self.config.supported_output_languages,
        }


class BatchTranslationPipeline(TranslationPipeline):
    """Pipeline optimized for batch processing."""

    async def translate_batch(
        self,
        items: list[dict],
        parallel_workers: int = 4,
    ) -> list[dict]:
        """
        Translate multiple items in parallel.

        Args:
            items: List of dicts with 'audio' or 'video' keys
            parallel_workers: Number of parallel workers

        Returns:
            List of translation results
        """
        semaphore = asyncio.Semaphore(parallel_workers)

        async def process_item(item: dict) -> dict:
            async with semaphore:
                if "video" in item:
                    return await self.translate_video(
                        video=item["video"],
                        source_lang=item.get("source_lang"),
                        target_lang=item.get("target_lang"),
                        speaker_id=item.get("speaker_id"),
                        output_path=item.get("output_path"),
                    )
                else:
                    return await self.translate_audio(
                        audio=item["audio"],
                        source_lang=item.get("source_lang"),
                        target_lang=item.get("target_lang"),
                        speaker_id=item.get("speaker_id"),
                    )

        results = await asyncio.gather(
            *[process_item(item) for item in items],
            return_exceptions=True,
        )

        return [r if not isinstance(r, Exception) else {"error": str(r)} for r in results]
