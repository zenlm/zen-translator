"""Tests for translation pipeline."""


class TestTranslationPipeline:
    """Tests for TranslationPipeline."""

    def test_pipeline_initialization(self, translator_config):
        """Test pipeline can be initialized."""
        from zen_translator.pipeline import TranslationPipeline

        pipeline = TranslationPipeline(translator_config)

        assert pipeline.config == translator_config
        assert pipeline.translator is not None
        assert pipeline.voice_cloner is not None
        assert pipeline._loaded is False

    def test_get_supported_languages(self, translator_config):
        """Test getting supported languages."""
        from zen_translator.pipeline import TranslationPipeline

        pipeline = TranslationPipeline(translator_config)
        languages = pipeline.get_supported_languages()

        assert "input" in languages
        assert "output" in languages
        assert len(languages["input"]) >= 18
        assert len(languages["output"]) == 10


class TestBatchTranslationPipeline:
    """Tests for BatchTranslationPipeline."""

    def test_batch_pipeline_initialization(self, translator_config):
        """Test batch pipeline can be initialized."""
        from zen_translator.pipeline import BatchTranslationPipeline

        pipeline = BatchTranslationPipeline(translator_config)

        assert pipeline.config == translator_config


class TestPipelineConfig:
    """Tests for pipeline configuration options."""

    def test_default_config(self):
        """Test default pipeline configuration."""
        from zen_translator import TranslatorConfig

        config = TranslatorConfig()

        assert config.qwen3_omni_model == "Qwen/Qwen3-Omni-30B-A3B-Instruct"
        assert config.cosyvoice_model == "FunAudioLLM/CosyVoice2-0.5B"
        assert config.wav2lip_model == "numz/wav2lip_studio"

    def test_custom_model_paths(self):
        """Test custom model path configuration."""
        from zen_translator import TranslatorConfig

        config = TranslatorConfig(
            qwen3_omni_model="./local/qwen3-omni",
            cosyvoice_model="./local/cosyvoice",
        )

        assert config.qwen3_omni_model == "./local/qwen3-omni"
        assert config.cosyvoice_model == "./local/cosyvoice"
