"""
Wav2Lip lip synchronization module.

Generates accurate lip movements synchronized with translated audio.
Optimized for real-time video dubbing applications.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from ..config import TranslatorConfig

logger = logging.getLogger(__name__)


class Wav2LipSync:
    """Lip synchronization using Wav2Lip."""

    # Quality presets
    QUALITY_PRESETS = {
        "fast": {
            "resize_factor": 2,
            "face_det_batch_size": 16,
            "wav2lip_batch_size": 128,
        },
        "balanced": {
            "resize_factor": 1,
            "face_det_batch_size": 8,
            "wav2lip_batch_size": 64,
        },
        "quality": {
            "resize_factor": 1,
            "face_det_batch_size": 4,
            "wav2lip_batch_size": 32,
        },
    }

    def __init__(self, config: TranslatorConfig):
        self.config = config
        self.model = None
        self.face_detector = None
        self._loaded = False

        self.preset = self.QUALITY_PRESETS[config.lip_sync_quality]

    def load(self) -> None:
        """Load Wav2Lip model and face detector."""
        if self._loaded:
            return

        logger.info(f"Loading Wav2Lip from {self.config.wav2lip_model}")

        try:
            # Load face detection model
            self._load_face_detector()

            # Load Wav2Lip model
            self._load_wav2lip_model()

            self._loaded = True
            logger.info("Wav2Lip loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Wav2Lip: {e}")
            raise

    def _load_face_detector(self) -> None:
        """Load face detection model."""
        try:
            import face_alignment

            self.face_detector = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D,
                device=self.config.device,
                flip_input=False,
            )
        except ImportError:
            logger.warning("face_alignment not installed, using OpenCV fallback")
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

    def _load_wav2lip_model(self) -> None:
        """Load Wav2Lip synthesis model."""
        from huggingface_hub import hf_hub_download

        # Download model checkpoint
        model_path = hf_hub_download(
            repo_id=self.config.wav2lip_model,
            filename="wav2lip.pth",
            cache_dir=self.config.model_cache_dir,
        )

        # Load model architecture
        from .wav2lip_model import Wav2Lip as Wav2LipModel

        self.model = Wav2LipModel()
        checkpoint = torch.load(model_path, map_location=self.config.device)

        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.config.device)
        self.model.eval()

    def unload(self) -> None:
        """Unload models to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.face_detector is not None:
            del self.face_detector
            self.face_detector = None
        self._loaded = False
        torch.cuda.empty_cache()

    async def sync_video(
        self,
        video: Path | str | np.ndarray,
        audio: Path | str | np.ndarray,
        output_path: Path | None = None,
        audio_sample_rate: int = 16000,
    ) -> dict:
        """
        Synchronize video lip movements with audio.

        Args:
            video: Input video (path or frames array)
            audio: Translated audio (path or numpy array)
            output_path: Optional output video path
            audio_sample_rate: Sample rate of audio

        Returns:
            dict with output_path or video_frames
        """
        if not self._loaded:
            self.load()

        logger.info("Starting lip synchronization...")

        # Load video frames
        if isinstance(video, (str, Path)):
            frames, video_fps = self._load_video(str(video))
        else:
            frames = video
            video_fps = 25  # Default FPS

        # Load audio
        if isinstance(audio, (str, Path)):
            audio_array = self._load_audio(str(audio), audio_sample_rate)
        else:
            audio_array = audio

        # Detect faces in frames
        face_coords = self._detect_faces(frames)

        # Generate mel spectrogram from audio
        mel = self._audio_to_mel(audio_array, audio_sample_rate)

        # Generate lip-synced frames
        synced_frames = self._generate_lip_sync(frames, face_coords, mel)

        # Save or return result
        if output_path:
            self._save_video(synced_frames, audio_array, audio_sample_rate, video_fps, output_path)
            return {"output_path": str(output_path), "frame_count": len(synced_frames)}
        else:
            return {"video_frames": synced_frames, "fps": video_fps}

    async def sync_frame(
        self,
        frame: np.ndarray,
        audio_chunk: np.ndarray,
        face_coords: tuple | None = None,
    ) -> np.ndarray:
        """
        Synchronize a single frame with audio chunk.

        For real-time streaming applications.
        """
        if not self._loaded:
            self.load()

        # Detect face if coords not provided
        if face_coords is None:
            face_coords = self._detect_face_single(frame)

        if face_coords is None:
            return frame  # No face detected, return original

        # Generate mel for audio chunk
        mel = self._audio_to_mel(audio_chunk, sample_rate=16000)

        # Sync single frame
        synced_frame = self._sync_single_frame(frame, face_coords, mel)

        return synced_frame

    def _load_video(self, video_path: str) -> tuple[list[np.ndarray], float]:
        """Load video frames."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        return frames, fps

    def _load_audio(self, audio_path: str, target_sr: int) -> np.ndarray:
        """Load audio file."""
        import librosa

        audio, _ = librosa.load(audio_path, sr=target_sr)
        return audio

    def _detect_faces(self, frames: list[np.ndarray]) -> list[tuple | None]:
        """Detect faces in all frames."""
        face_coords = []

        for frame in frames:
            coords = self._detect_face_single(frame)
            face_coords.append(coords)

        # Interpolate missing detections
        face_coords = self._interpolate_missing_faces(face_coords)

        return face_coords

    def _detect_face_single(self, frame: np.ndarray) -> tuple | None:
        """Detect face in a single frame."""
        if hasattr(self.face_detector, "get_landmarks"):
            # face_alignment library
            landmarks = self.face_detector.get_landmarks(frame)
            if landmarks is None or len(landmarks) == 0:
                return None

            # Get bounding box from landmarks
            landmarks = landmarks[0]
            x_min, y_min = landmarks.min(axis=0).astype(int)
            x_max, y_max = landmarks.max(axis=0).astype(int)

            # Add padding
            padding = int(0.2 * (x_max - x_min))
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(frame.shape[1], x_max + padding)
            y_max = min(frame.shape[0], y_max + padding)

            return (x_min, y_min, x_max, y_max)
        else:
            # OpenCV fallback
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.1, 4)

            if len(faces) == 0:
                return None

            x, y, w, h = faces[0]
            return (x, y, x + w, y + h)

    def _interpolate_missing_faces(
        self,
        face_coords: list[tuple | None],
    ) -> list[tuple | None]:
        """Interpolate missing face detections."""
        # Find first and last valid detection
        valid_indices = [i for i, c in enumerate(face_coords) if c is not None]

        if not valid_indices:
            return face_coords

        result = face_coords.copy()

        # Forward fill
        last_valid = None
        for i, coords in enumerate(result):
            if coords is not None:
                last_valid = coords
            elif last_valid is not None:
                result[i] = last_valid

        return result

    def _audio_to_mel(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Convert audio to mel spectrogram."""
        import librosa

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_mels=80,
            n_fft=800,
            hop_length=200,
            win_length=800,
        )
        mel = librosa.power_to_db(mel, ref=np.max)

        return mel.T  # Transpose for time-first format

    def _generate_lip_sync(
        self,
        frames: list[np.ndarray],
        face_coords: list[tuple],
        mel: np.ndarray,
    ) -> list[np.ndarray]:
        """Generate lip-synced frames using Wav2Lip."""
        batch_size = self.preset["wav2lip_batch_size"]
        synced_frames = []

        # Calculate mel frames per video frame
        mel_idx_multiplier = len(mel) / len(frames)

        for batch_start in range(0, len(frames), batch_size):
            batch_end = min(batch_start + batch_size, len(frames))
            batch_frames = frames[batch_start:batch_end]
            batch_coords = face_coords[batch_start:batch_end]

            # Get corresponding mel frames
            mel_batch = []
            for i in range(batch_start, batch_end):
                mel_idx = int(i * mel_idx_multiplier)
                mel_window = mel[max(0, mel_idx - 8) : mel_idx + 8]

                # Pad if necessary
                if len(mel_window) < 16:
                    padding = np.zeros((16 - len(mel_window), mel.shape[1]))
                    mel_window = np.vstack([mel_window, padding])

                mel_batch.append(mel_window[:16])

            # Process batch
            batch_synced = self._process_batch(batch_frames, batch_coords, mel_batch)
            synced_frames.extend(batch_synced)

        return synced_frames

    def _process_batch(
        self,
        frames: list[np.ndarray],
        coords: list[tuple],
        mel_batch: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Process a batch of frames through Wav2Lip."""
        img_size = 96  # Wav2Lip face size

        # Prepare face crops
        face_crops = []
        for frame, coord in zip(frames, coords):
            if coord is None:
                face_crops.append(np.zeros((img_size, img_size, 3), dtype=np.uint8))
            else:
                x1, y1, x2, y2 = coord
                face = frame[y1:y2, x1:x2]
                face = cv2.resize(face, (img_size, img_size))
                face_crops.append(face)

        # Convert to tensors
        face_tensor = torch.FloatTensor(np.array(face_crops)).permute(0, 3, 1, 2) / 255.0
        mel_tensor = torch.FloatTensor(np.array(mel_batch))

        face_tensor = face_tensor.to(self.config.device)
        mel_tensor = mel_tensor.to(self.config.device)

        # Generate synced faces
        with torch.no_grad():
            synced_faces = self.model(mel_tensor, face_tensor)

        synced_faces = synced_faces.permute(0, 2, 3, 1).cpu().numpy() * 255
        synced_faces = synced_faces.astype(np.uint8)

        # Paste synced faces back into frames
        result_frames = []
        for i, (frame, coord) in enumerate(zip(frames, coords)):
            if coord is None:
                result_frames.append(frame)
                continue

            x1, y1, x2, y2 = coord
            synced_face = cv2.resize(synced_faces[i], (x2 - x1, y2 - y1))

            result = frame.copy()
            result[y1:y2, x1:x2] = synced_face
            result_frames.append(result)

        return result_frames

    def _sync_single_frame(
        self,
        frame: np.ndarray,
        face_coords: tuple,
        mel: np.ndarray,
    ) -> np.ndarray:
        """Sync a single frame for real-time streaming."""
        return self._process_batch([frame], [face_coords], [mel[:16]])[0]

    def _save_video(
        self,
        frames: list[np.ndarray],
        audio: np.ndarray,
        audio_sr: int,
        fps: float,
        output_path: Path,
    ) -> None:
        """Save lip-synced video with audio."""
        import subprocess
        import tempfile

        # Save frames to temp video
        temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_video.name, fourcc, fps, (width, height))

        for frame in frames:
            writer.write(frame)
        writer.release()

        # Save audio
        import soundfile as sf

        sf.write(temp_audio.name, audio, audio_sr)

        # Combine video and audio with ffmpeg
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                temp_video.name,
                "-i",
                temp_audio.name,
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-strict",
                "experimental",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )

        # Cleanup
        Path(temp_video.name).unlink()
        Path(temp_audio.name).unlink()
