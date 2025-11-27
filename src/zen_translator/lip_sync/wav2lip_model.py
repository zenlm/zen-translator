"""
Wav2Lip neural network architecture.

Based on the original Wav2Lip paper:
"A Lip Sync Expert Is All You Need for Speech to Lip Generation In The Wild"
https://arxiv.org/abs/2008.10010
"""

import torch
import torch.nn as nn


class Conv2d(nn.Module):
    """2D convolution with weight standardization option."""

    def __init__(
        self,
        cin: int,
        cout: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        residual: bool = False,
    ):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout),
        )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class ConvTranspose2d(nn.Module):
    """Transposed 2D convolution for upsampling."""

    def __init__(
        self,
        cin: int,
        cout: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
    ):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(cout),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_block(x)
        return self.act(out)


class ResBlock(nn.Module):
    """Residual block with two convolutions."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            Conv2d(out_channels, out_channels, kernel_size=3, padding=1, residual=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AudioEncoder(nn.Module):
    """Encoder for mel spectrogram audio features."""

    def __init__(self):
        super().__init__()

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, audio_sequences: torch.Tensor) -> torch.Tensor:
        # audio_sequences: (batch_size, T, 1, 80, 16)
        batch_size = audio_sequences.size(0)
        audio_sequences = audio_sequences.view(
            -1, 1, audio_sequences.size(3), audio_sequences.size(4)
        )
        audio_embedding = self.audio_encoder(audio_sequences)
        audio_embedding = audio_embedding.view(batch_size, -1, 512, 1, 1)
        return audio_embedding


class FaceEncoder(nn.Module):
    """Encoder for face image features."""

    def __init__(self):
        super().__init__()

        self.face_encoder_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    Conv2d(6, 16, kernel_size=7, stride=1, padding=3),
                ),  # 96, 96
                nn.Sequential(
                    Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                    Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                ),  # 48, 48
                nn.Sequential(
                    Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                ),  # 24, 24
                nn.Sequential(
                    Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                ),  # 12, 12
                nn.Sequential(
                    Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                    Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                ),  # 6, 6
                nn.Sequential(
                    Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                    Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                ),  # 3, 3
                nn.Sequential(
                    Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
                    Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                ),  # 1, 1
            ]
        )

    def forward(self, face_sequences: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        x = face_sequences
        for block in self.face_encoder_blocks:
            x = block(x)
            feats.append(x)
        return feats


class FaceDecoder(nn.Module):
    """Decoder to generate lip-synced face."""

    def __init__(self):
        super().__init__()

        self.face_decoder_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                ),
                nn.Sequential(
                    ConvTranspose2d(1024, 512, kernel_size=3, stride=1, padding=0),
                    Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                ),  # 3, 3
                nn.Sequential(
                    ConvTranspose2d(
                        1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                ),  # 6, 6
                nn.Sequential(
                    ConvTranspose2d(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
                    Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                ),  # 12, 12
                nn.Sequential(
                    ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                    Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                ),  # 24, 24
                nn.Sequential(
                    ConvTranspose2d(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                    Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                ),  # 48, 48
                nn.Sequential(
                    ConvTranspose2d(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                    Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                ),  # 96, 96
            ]
        )

        self.output_block = nn.Sequential(
            Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(
        self, audio_embedding: torch.Tensor, face_features: list[torch.Tensor]
    ) -> torch.Tensor:
        x = audio_embedding
        for i, block in enumerate(self.face_decoder_blocks):
            x = block(x)
            if i < len(face_features):
                # Skip connection from encoder
                skip = face_features[-(i + 1)]
                x = torch.cat([x, skip], dim=1)

        x = self.output_block(x)
        return x


class Wav2Lip(nn.Module):
    """
    Wav2Lip model for lip synchronization.

    Takes mel spectrogram audio features and face images,
    generates lip-synced face images.
    """

    def __init__(self):
        super().__init__()

        self.audio_encoder = AudioEncoder()
        self.face_encoder = FaceEncoder()
        self.face_decoder = FaceDecoder()

    def forward(
        self,
        audio_sequences: torch.Tensor,
        face_sequences: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate lip-synced faces.

        Args:
            audio_sequences: Mel spectrogram features (B, T, 1, 80, 16)
            face_sequences: Face images (B, 6, 96, 96) - 6 channels for half face + reference

        Returns:
            Generated face images (B, 3, 96, 96)
        """
        # Encode audio
        audio_embedding = self.audio_encoder(audio_sequences)
        audio_embedding = audio_embedding.squeeze(1)  # (B, 512, 1, 1)

        # Encode face
        face_features = self.face_encoder(face_sequences)

        # Decode to generate lip-synced face
        output = self.face_decoder(audio_embedding, face_features)

        return output


class Wav2LipGAN(Wav2Lip):
    """Wav2Lip with GAN discriminator for higher quality."""

    def __init__(self):
        super().__init__()

        # Discriminator for sync detection
        self.sync_discriminator = SyncDiscriminator()

    def sync_loss(
        self,
        mel: torch.Tensor,
        generated_face: torch.Tensor,
        real_face: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute sync discriminator loss."""
        # Real sync
        real_sync = self.sync_discriminator(mel, real_face)
        # Fake sync
        fake_sync = self.sync_discriminator(mel, generated_face)

        return real_sync, fake_sync


class SyncDiscriminator(nn.Module):
    """Discriminator for audio-visual sync detection."""

    def __init__(self):
        super().__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(3, 32, kernel_size=7, stride=1, padding=3),
            Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, mel: torch.Tensor, face: torch.Tensor) -> torch.Tensor:
        face_embedding = self.face_encoder(face)
        face_embedding = face_embedding.view(face.size(0), -1)

        audio_embedding = self.audio_encoder(mel.unsqueeze(1))
        audio_embedding = audio_embedding.view(mel.size(0), -1)

        combined = torch.cat([face_embedding, audio_embedding], dim=1)
        return self.fc(combined)
