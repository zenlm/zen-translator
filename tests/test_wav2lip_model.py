"""Tests for Wav2Lip model architecture."""

import torch


class TestWav2LipModel:
    """Tests for Wav2Lip neural network."""

    def test_model_initialization(self):
        """Test model can be initialized."""
        from zen_translator.lip_sync.wav2lip_model import Wav2Lip

        model = Wav2Lip()

        assert model.audio_encoder is not None
        assert model.face_encoder is not None
        assert model.face_decoder is not None

    def test_model_forward_shape(self):
        """Test model forward pass produces correct output shape."""
        from zen_translator.lip_sync.wav2lip_model import Wav2Lip

        model = Wav2Lip()
        model.eval()

        # Create dummy inputs
        batch_size = 2
        mel_length = 16
        mel_channels = 80

        # Audio: (B, T, 1, 80, 16) -> mel spectrogram windows
        audio = torch.randn(batch_size, 1, 1, mel_channels, mel_length)

        # Face: (B, 6, 96, 96) -> half face + reference
        face = torch.randn(batch_size, 6, 96, 96)

        with torch.no_grad():
            output = model(audio, face)

        # Output should be (B, 3, 96, 96)
        assert output.shape == (batch_size, 3, 96, 96)

    def test_audio_encoder(self):
        """Test audio encoder produces correct embedding."""
        from zen_translator.lip_sync.wav2lip_model import AudioEncoder

        encoder = AudioEncoder()
        encoder.eval()

        batch_size = 2
        audio = torch.randn(batch_size, 1, 1, 80, 16)

        with torch.no_grad():
            embedding = encoder(audio)

        # Should produce 512-dim embedding
        assert embedding.shape[-3] == 512

    def test_face_encoder(self):
        """Test face encoder produces feature hierarchy."""
        from zen_translator.lip_sync.wav2lip_model import FaceEncoder

        encoder = FaceEncoder()
        encoder.eval()

        batch_size = 2
        face = torch.randn(batch_size, 6, 96, 96)

        with torch.no_grad():
            features = encoder(face)

        # Should produce 7 feature maps (one per block)
        assert len(features) == 7


class TestConvBlocks:
    """Tests for convolution building blocks."""

    def test_conv2d_block(self):
        """Test Conv2d block."""
        from zen_translator.lip_sync.wav2lip_model import Conv2d

        block = Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        x = torch.randn(1, 3, 64, 64)

        out = block(x)

        assert out.shape == (1, 32, 64, 64)

    def test_conv2d_residual(self):
        """Test Conv2d with residual connection."""
        from zen_translator.lip_sync.wav2lip_model import Conv2d

        block = Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)
        x = torch.randn(1, 32, 64, 64)

        out = block(x)

        # With residual, output should be different from non-residual
        assert out.shape == (1, 32, 64, 64)

    def test_transpose_conv2d(self):
        """Test ConvTranspose2d block."""
        from zen_translator.lip_sync.wav2lip_model import ConvTranspose2d

        block = ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        x = torch.randn(1, 32, 32, 32)

        out = block(x)

        # Should upsample by factor of 2
        assert out.shape == (1, 16, 64, 64)


class TestSyncDiscriminator:
    """Tests for sync discriminator."""

    def test_discriminator_output(self):
        """Test sync discriminator produces probability."""
        from zen_translator.lip_sync.wav2lip_model import SyncDiscriminator

        discriminator = SyncDiscriminator()
        discriminator.eval()

        batch_size = 2
        mel = torch.randn(batch_size, 80, 16)
        face = torch.randn(batch_size, 3, 96, 96)

        with torch.no_grad():
            output = discriminator(mel, face)

        # Should produce sync probability
        assert output.shape == (batch_size, 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)
