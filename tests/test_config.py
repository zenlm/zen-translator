"""Tests for configuration module."""


class TestTranslatorConfig:
    """Tests for TranslatorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from zen_translator.config import TranslatorConfig

        config = TranslatorConfig()

        assert config.target_language == "en"
        assert config.device == "cuda"
        assert config.dtype == "bfloat16"
        assert config.enable_lip_sync is True
        assert config.voice_reference_seconds == 3.0

    def test_config_from_env(self, monkeypatch):
        """Test configuration from environment variables."""
        from zen_translator.config import TranslatorConfig

        monkeypatch.setenv("ZEN_TRANSLATOR_TARGET_LANGUAGE", "es")
        monkeypatch.setenv("ZEN_TRANSLATOR_DEVICE", "cpu")

        config = TranslatorConfig()

        assert config.target_language == "es"
        assert config.device == "cpu"

    def test_supported_languages(self):
        """Test supported language lists."""
        from zen_translator.config import TranslatorConfig

        config = TranslatorConfig()

        # Check input languages
        assert "en" in config.supported_input_languages
        assert "zh" in config.supported_input_languages
        assert "ja" in config.supported_input_languages
        assert "yue" in config.supported_input_languages  # Cantonese

        # Check output languages
        assert "en" in config.supported_output_languages
        assert "zh" in config.supported_output_languages
        assert len(config.supported_output_languages) == 10

    def test_lip_sync_quality_options(self):
        """Test lip sync quality options."""
        from zen_translator.config import TranslatorConfig

        for quality in ["fast", "balanced", "quality"]:
            config = TranslatorConfig(lip_sync_quality=quality)
            assert config.lip_sync_quality == quality


class TestNewsAnchorConfig:
    """Tests for NewsAnchorConfig."""

    def test_default_config(self):
        """Test default news anchor config."""
        from zen_translator.config import NewsAnchorConfig

        config = NewsAnchorConfig()

        assert config.min_clip_duration == 5.0
        assert config.max_clip_duration == 30.0
        assert len(config.target_anchors) > 0

    def test_training_settings(self):
        """Test training hyperparameters."""
        from zen_translator.config import NewsAnchorConfig

        config = NewsAnchorConfig()

        assert config.batch_size == 4
        assert config.learning_rate == 2e-5
        assert config.num_epochs == 3
