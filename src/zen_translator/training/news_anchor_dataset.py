"""
News anchor dataset collection and processing pipeline.

Collects, processes, and prepares news anchor audio/video data
for finetuning Zen Translator for accurate broadcast translation.
"""

import json
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..config import NewsAnchorConfig

logger = logging.getLogger(__name__)


@dataclass
class NewsAnchorSample:
    """A single news anchor audio/video sample."""

    anchor_id: str
    audio_path: Path
    video_path: Path | None
    transcript: str
    language: str
    duration_seconds: float
    news_domain: str
    timestamp: datetime
    source_url: str | None = None

    def to_dict(self) -> dict:
        return {
            "anchor_id": self.anchor_id,
            "audio_path": str(self.audio_path),
            "video_path": str(self.video_path) if self.video_path else None,
            "transcript": self.transcript,
            "language": self.language,
            "duration_seconds": self.duration_seconds,
            "news_domain": self.news_domain,
            "timestamp": self.timestamp.isoformat(),
            "source_url": self.source_url,
        }


class NewsAnchorDatasetBuilder:
    """
    Builds training datasets from news anchor recordings.

    Pipeline:
    1. Collect audio/video from news sources
    2. Extract and transcribe speech
    3. Segment into training samples
    4. Create translation pairs
    5. Export in ms-swift format
    """

    def __init__(self, config: NewsAnchorConfig):
        self.config = config
        self.samples: list[NewsAnchorSample] = []

    async def collect_from_youtube(
        self,
        channel_urls: list[str],
        max_videos_per_channel: int = 10,
    ) -> AsyncIterator[NewsAnchorSample]:
        """
        Collect news anchor data from YouTube channels.

        Supports channels like:
        - CNN, BBC News, NHK World, DW News, etc.
        """
        try:
            import yt_dlp
        except ImportError:
            logger.error("yt-dlp not installed. Run: pip install yt-dlp")
            return

        output_dir = self.config.dataset_dir / "raw" / "youtube"
        output_dir.mkdir(parents=True, exist_ok=True)

        ydl_opts = {
            "format": "bestvideo[height<=720]+bestaudio/best[height<=720]",
            "outtmpl": str(output_dir / "%(channel)s/%(id)s.%(ext)s"),
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en", "zh", "ja", "ko", "es", "fr", "de"],
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                },
            ],
            "max_downloads": max_videos_per_channel,
        }

        for channel_url in channel_urls:
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(channel_url, download=True)

                    for entry in info.get("entries", []):
                        if entry is None:
                            continue

                        video_id = entry["id"]
                        channel_name = entry.get("channel", "unknown")

                        # Find downloaded files
                        audio_path = output_dir / channel_name / f"{video_id}.wav"
                        video_path = output_dir / channel_name / f"{video_id}.mp4"

                        if not audio_path.exists():
                            continue

                        # Get transcript from subtitles
                        transcript = await self._extract_transcript(
                            entry, output_dir / channel_name
                        )

                        sample = NewsAnchorSample(
                            anchor_id=channel_name.lower().replace(" ", "_"),
                            audio_path=audio_path,
                            video_path=video_path if video_path.exists() else None,
                            transcript=transcript,
                            language=entry.get("language", "en"),
                            duration_seconds=entry.get("duration", 0),
                            news_domain=self._detect_news_domain(entry.get("title", "")),
                            timestamp=datetime.now(),
                            source_url=entry.get("webpage_url"),
                        )

                        self.samples.append(sample)
                        yield sample

            except Exception as e:
                logger.error(f"Error collecting from {channel_url}: {e}")

    async def _extract_transcript(self, entry: dict, output_dir: Path) -> str:
        """Extract transcript from video subtitles."""
        video_id = entry["id"]

        # Try different subtitle formats
        for ext in [".en.vtt", ".en.srt", ".vtt", ".srt"]:
            sub_path = output_dir / f"{video_id}{ext}"
            if sub_path.exists():
                return self._parse_subtitle_file(sub_path)

        # Fallback to auto-generated transcript
        return entry.get("description", "")[:500]

    def _parse_subtitle_file(self, path: Path) -> str:
        """Parse VTT or SRT subtitle file."""
        content = path.read_text()

        # Remove timing information and formatting
        lines = []
        for line in content.split("\n"):
            # Skip timing lines
            if re.match(r"^\d+:\d+", line) or re.match(r"^\d+$", line):
                continue
            # Skip WebVTT header
            if line.startswith("WEBVTT") or line.startswith("Kind:"):
                continue
            # Clean HTML tags
            line = re.sub(r"<[^>]+>", "", line)
            if line.strip():
                lines.append(line.strip())

        return " ".join(lines)

    def _detect_news_domain(self, title: str) -> str:
        """Detect news domain from video title."""
        title_lower = title.lower()

        domain_keywords = {
            "politics": ["election", "vote", "congress", "parliament", "president", "minister"],
            "economics": ["economy", "market", "stock", "trade", "inflation", "gdp"],
            "technology": ["tech", "ai", "software", "startup", "digital", "cyber"],
            "sports": ["game", "match", "championship", "olympics", "team", "player"],
            "weather": ["weather", "storm", "hurricane", "temperature", "forecast"],
            "breaking_news": ["breaking", "urgent", "just in", "developing"],
            "international": ["world", "global", "international", "foreign"],
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in title_lower for kw in keywords):
                return domain

        return "general"

    async def segment_samples(
        self,
        min_duration: float = 5.0,
        max_duration: float = 30.0,
    ) -> list[NewsAnchorSample]:
        """Segment long recordings into training-sized chunks."""
        import librosa

        segmented = []

        for sample in self.samples:
            if sample.duration_seconds <= max_duration:
                if sample.duration_seconds >= min_duration:
                    segmented.append(sample)
                continue

            # Load audio
            audio, sr = librosa.load(str(sample.audio_path), sr=16000)

            # Split into chunks
            chunk_samples = int(max_duration * sr)
            hop_samples = int(chunk_samples * 0.8)  # 20% overlap

            for i, start in enumerate(range(0, len(audio) - chunk_samples, hop_samples)):
                chunk = audio[start : start + chunk_samples]

                # Save chunk
                chunk_path = sample.audio_path.parent / f"{sample.audio_path.stem}_chunk{i}.wav"
                import soundfile as sf

                sf.write(str(chunk_path), chunk, sr)

                # Create new sample
                chunk_sample = NewsAnchorSample(
                    anchor_id=sample.anchor_id,
                    audio_path=chunk_path,
                    video_path=None,  # Video segmentation is more complex
                    transcript=f"[Chunk {i}] {sample.transcript}",  # Would need alignment
                    language=sample.language,
                    duration_seconds=max_duration,
                    news_domain=sample.news_domain,
                    timestamp=sample.timestamp,
                    source_url=sample.source_url,
                )
                segmented.append(chunk_sample)

        self.samples = segmented
        return segmented

    async def create_translation_pairs(
        self,
        target_languages: list[str] = ["en", "zh", "ja", "es"],
    ) -> list[dict]:
        """Create translation pairs for training."""
        from ..config import TranslatorConfig
        from ..translation import Qwen3OmniTranslator

        config = TranslatorConfig()
        translator = Qwen3OmniTranslator(config)
        translator.load()

        pairs = []

        for sample in self.samples:
            for target_lang in target_languages:
                if target_lang == sample.language:
                    continue

                # Translate transcript
                try:
                    # For actual training, we'd use actual audio translation
                    # Here we show the data format
                    pairs.append(
                        {
                            "conversations": [
                                {
                                    "role": "system",
                                    "content": f"You are Zen Translator. Translate the speech to {target_lang}.",
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "audio", "audio": str(sample.audio_path)},
                                        {"type": "text", "text": f"Translate to {target_lang}."},
                                    ],
                                },
                                {
                                    "role": "assistant",
                                    "content": f"[{target_lang}] {sample.transcript}",  # Placeholder
                                },
                            ],
                            "metadata": {
                                "anchor_id": sample.anchor_id,
                                "source_lang": sample.language,
                                "target_lang": target_lang,
                                "domain": sample.news_domain,
                            },
                        }
                    )
                except Exception as e:
                    logger.error(f"Error creating pair: {e}")

        return pairs

    async def export_dataset(
        self,
        output_path: Path,
        format: str = "jsonl",
        split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> dict[str, Path]:
        """
        Export dataset for ms-swift training.

        Returns paths to train/val/test splits.
        """
        import random

        pairs = await self.create_translation_pairs()
        random.shuffle(pairs)

        n = len(pairs)
        train_end = int(n * split_ratio[0])
        val_end = train_end + int(n * split_ratio[1])

        splits = {
            "train": pairs[:train_end],
            "val": pairs[train_end:val_end],
            "test": pairs[val_end:],
        }

        output_path.mkdir(parents=True, exist_ok=True)
        paths = {}

        for split_name, split_data in splits.items():
            split_path = output_path / f"{split_name}.jsonl"

            with open(split_path, "w") as f:
                for item in split_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            paths[split_name] = split_path
            logger.info(f"Exported {len(split_data)} samples to {split_path}")

        # Save metadata
        metadata = {
            "total_samples": len(pairs),
            "splits": {k: len(v) for k, v in splits.items()},
            "anchors": list(set(s.anchor_id for s in self.samples)),
            "languages": list(set(s.language for s in self.samples)),
            "domains": list(set(s.news_domain for s in self.samples)),
            "created": datetime.now().isoformat(),
        }

        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return paths


# Predefined news channel URLs for data collection
NEWS_CHANNELS = {
    "cnn": "https://www.youtube.com/@CNN",
    "bbc": "https://www.youtube.com/@BBCNews",
    "nhk": "https://www.youtube.com/@NHKWORLDJAPAN",
    "dw": "https://www.youtube.com/@DWNews",
    "france24_en": "https://www.youtube.com/@FRANCE24English",
    "aljazeera": "https://www.youtube.com/@AlJazeeraEnglish",
    "sky": "https://www.youtube.com/@SkyNews",
    "reuters": "https://www.youtube.com/@Reuters",
    "ap": "https://www.youtube.com/@AssociatedPress",
    "bloomberg": "https://www.youtube.com/@BloombergTelevision",
    # Non-English channels
    "cctv": "https://www.youtube.com/@CCTVVideoNewsAgency",
    "nhk_ja": "https://www.youtube.com/@NHK",
    "tbs_ja": "https://www.youtube.com/@tbsnewsdig",
    "kbs_ko": "https://www.youtube.com/@KBSNews",
    "tvn_ko": "https://www.youtube.com/@tvaborigen",
}


async def build_news_anchor_dataset(
    output_dir: Path,
    channels: list[str] | None = None,
    max_videos_per_channel: int = 10,
) -> Path:
    """
    Convenience function to build a news anchor dataset.

    Args:
        output_dir: Output directory for dataset
        channels: List of channel keys from NEWS_CHANNELS
        max_videos_per_channel: Max videos to download per channel

    Returns:
        Path to the created dataset
    """
    from ..config import NewsAnchorConfig

    config = NewsAnchorConfig()
    config.dataset_dir = output_dir

    builder = NewsAnchorDatasetBuilder(config)

    # Select channels
    if channels is None:
        channels = ["cnn", "bbc", "nhk", "dw"]

    channel_urls = [NEWS_CHANNELS[c] for c in channels if c in NEWS_CHANNELS]

    # Collect data
    logger.info(f"Collecting from {len(channel_urls)} channels...")
    async for sample in builder.collect_from_youtube(channel_urls, max_videos_per_channel):
        logger.info(f"Collected: {sample.anchor_id} - {sample.duration_seconds:.1f}s")

    # Segment
    logger.info("Segmenting samples...")
    await builder.segment_samples()

    # Export
    logger.info("Exporting dataset...")
    await builder.export_dataset(output_dir / "processed")

    return output_dir / "processed"
