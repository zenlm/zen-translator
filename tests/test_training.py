"""Tests for training infrastructure."""


class TestSwiftConfig:
    """Tests for ms-swift training configuration."""

    def test_default_config(self):
        """Test default training config."""
        from zen_translator.training import SwiftTrainingConfig

        config = SwiftTrainingConfig()

        assert config.model_type == "qwen3-omni"
        assert config.train_type == "lora"
        assert config.lora_rank == 64
        assert config.lora_alpha == 128

    def test_to_swift_args(self):
        """Test conversion to swift CLI arguments."""
        from zen_translator.training import SwiftTrainingConfig

        config = SwiftTrainingConfig()
        args = config.to_swift_args()

        assert "--model_type=qwen3-omni" in args
        assert "--train_type=lora" in args
        assert "--lora_rank=64" in args

    def test_to_yaml(self, tmp_path):
        """Test YAML export."""
        from zen_translator.training import SwiftTrainingConfig

        config = SwiftTrainingConfig()
        yaml_path = tmp_path / "config.yaml"

        config.to_yaml(yaml_path)

        assert yaml_path.exists()

        # Verify content
        import yaml

        with open(yaml_path) as f:
            saved = yaml.safe_load(f)

        assert saved["model"]["type"] == "qwen3-omni"
        assert saved["lora"]["rank"] == 64


class TestZenIdentityConfig:
    """Tests for Zen identity finetuning config."""

    def test_identity_system_prompt(self):
        """Test identity system prompt is set."""
        from zen_translator.training import ZenIdentityConfig

        config = ZenIdentityConfig()

        assert "Zen Translator" in config.system_prompt
        assert "Hanzo AI" in config.system_prompt


class TestNewsAnchorConfig:
    """Tests for news anchor training config."""

    def test_anchor_names(self):
        """Test anchor names are configured."""
        from zen_translator.training import NewsAnchorConfig

        config = NewsAnchorConfig()

        assert len(config.anchor_names) > 0
        assert "cnn" in config.anchor_names
        assert "bbc" in config.anchor_names

    def test_news_domains(self):
        """Test news domains are configured."""
        from zen_translator.training import NewsAnchorConfig

        config = NewsAnchorConfig()

        assert "politics" in config.news_domains
        assert "technology" in config.news_domains


class TestNewsChannels:
    """Tests for predefined news channels."""

    def test_channels_defined(self):
        """Test news channels are defined."""
        from zen_translator.training import NEWS_CHANNELS

        assert len(NEWS_CHANNELS) > 0
        assert "cnn" in NEWS_CHANNELS
        assert "bbc" in NEWS_CHANNELS
        assert "nhk" in NEWS_CHANNELS

    def test_channel_urls(self):
        """Test channel URLs are valid."""
        from zen_translator.training import NEWS_CHANNELS

        for name, url in NEWS_CHANNELS.items():
            assert url.startswith("https://")
            assert "youtube.com" in url


class TestCreateTrainingDataset:
    """Tests for dataset creation."""

    def test_create_jsonl_dataset(self, tmp_path):
        """Test JSONL dataset creation."""
        from zen_translator.training import create_training_dataset

        conversations = [
            {
                "conversations": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            }
        ]

        output_path = tmp_path / "train.jsonl"
        create_training_dataset(conversations, output_path, format="jsonl")

        assert output_path.exists()

        # Verify content
        import json

        with open(output_path) as f:
            lines = f.readlines()

        assert len(lines) == 1
        data = json.loads(lines[0])
        assert "conversations" in data
