"""Configuration for Zen Translator pipeline."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class TranslatorConfig(BaseSettings):
    """Configuration for the translation pipeline."""

    # Model paths
    qwen3_omni_model: str = Field(
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct", description="Qwen3-Omni model for translation"
    )
    cosyvoice_model: str = Field(
        default="FunAudioLLM/CosyVoice2-0.5B", description="CosyVoice model for voice cloning"
    )
    wav2lip_model: str = Field(
        default="numz/wav2lip_studio", description="Wav2Lip model for lip sync"
    )

    # Local model cache
    model_cache_dir: Path = Field(
        default=Path("./models"), description="Directory to cache downloaded models"
    )

    # Translation settings
    source_language: str = Field(default="auto", description="Source language (auto-detect)")
    target_language: str = Field(default="en", description="Target language for translation")

    # Supported languages
    # Input: 18 languages + 6 dialects
    supported_input_languages: list[str] = [
        "en",
        "zh",
        "ja",
        "ko",
        "es",
        "fr",
        "de",
        "it",
        "pt",
        "ru",
        "ar",
        "hi",
        "th",
        "vi",
        "id",
        "ms",
        "tr",
        "pl",
        # Dialects
        "yue",  # Cantonese
        "wuu",  # Shanghainese
        "hsn",  # Xiang
        "nan",  # Min Nan
        "hak",  # Hakka
        "cdo",  # Min Dong
    ]
    # Output: 10 languages
    supported_output_languages: list[str] = [
        "en",
        "zh",
        "ja",
        "ko",
        "es",
        "fr",
        "de",
        "it",
        "pt",
        "ru",
    ]

    # Voice cloning settings
    voice_reference_seconds: float = Field(
        default=3.0, description="Minimum seconds of reference audio for voice cloning"
    )
    preserve_emotion: bool = Field(
        default=True, description="Preserve speaker emotion in cloned voice"
    )
    preserve_inflection: bool = Field(
        default=True, description="Preserve speaker inflection patterns"
    )

    # Lip sync settings
    enable_lip_sync: bool = Field(default=True, description="Enable lip synchronization")
    lip_sync_quality: Literal["fast", "balanced", "quality"] = Field(
        default="balanced", description="Lip sync quality/speed tradeoff"
    )

    # Streaming settings
    streaming_chunk_ms: int = Field(
        default=200, description="Audio chunk size in milliseconds for streaming"
    )
    buffer_size_ms: int = Field(default=500, description="Buffer size for smoother playback")

    # Hardware settings
    device: Literal["cuda", "cpu", "mps"] = Field(
        default="cuda", description="Device to run models on"
    )
    dtype: Literal["float16", "bfloat16", "float32"] = Field(
        default="bfloat16", description="Model precision"
    )

    # Performance tuning
    use_flash_attention: bool = Field(default=True, description="Use Flash Attention 2")
    compile_model: bool = Field(default=False, description="Use torch.compile")

    # Finetuning settings (for news anchor voices)
    finetune_enabled: bool = Field(default=False, description="Enable finetuning mode")
    finetune_output_dir: Path = Field(
        default=Path("./outputs/finetune"), description="Output directory for finetuned models"
    )
    lora_rank: int = Field(default=64, description="LoRA rank for finetuning")
    lora_alpha: int = Field(default=128, description="LoRA alpha")

    model_config = {
        "env_prefix": "ZEN_TRANSLATOR_",
        "env_file": ".env",
    }


class NewsAnchorConfig(BaseSettings):
    """Configuration for news anchor voice training."""

    # Dataset settings
    dataset_dir: Path = Field(
        default=Path("./data/news_anchors"),
        description="Directory containing news anchor audio/video data",
    )
    min_clip_duration: float = Field(default=5.0, description="Minimum clip duration in seconds")
    max_clip_duration: float = Field(default=30.0, description="Maximum clip duration in seconds")

    # Target anchors (examples)
    target_anchors: list[str] = [
        "anderson_cooper",
        "rachel_maddow",
        "tucker_carlson",
        "don_lemon",
        "wolf_blitzer",
        "bbc_news",
        "cnn_international",
        "sky_news",
        "nhk_world",
        "dw_news",
    ]

    # Training settings
    batch_size: int = Field(default=4, description="Training batch size")
    gradient_accumulation_steps: int = Field(default=8, description="Gradient accumulation")
    learning_rate: float = Field(default=2e-5, description="Learning rate")
    num_epochs: int = Field(default=3, description="Number of training epochs")
    warmup_ratio: float = Field(default=0.1, description="Warmup ratio")

    # Data augmentation
    augment_noise: bool = Field(default=True, description="Add background noise augmentation")
    augment_speed: bool = Field(default=True, description="Speed variation augmentation")
    noise_levels: list[float] = [0.01, 0.02, 0.05]
    speed_factors: list[float] = [0.9, 0.95, 1.0, 1.05, 1.1]

    model_config = {
        "env_prefix": "ZEN_ANCHOR_",
    }
