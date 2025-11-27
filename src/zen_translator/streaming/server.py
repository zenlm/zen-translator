"""
Real-time streaming translation server.

Provides WebSocket and REST APIs for:
- Real-time audio translation
- Video translation with lip sync
- Voice cloning management
- WebRTC integration
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..config import TranslatorConfig
from ..pipeline import TranslationPipeline

logger = logging.getLogger(__name__)


class TranslationRequest(BaseModel):
    """Request for text-based translation."""

    text: str
    source_lang: str | None = None
    target_lang: str = "en"
    speaker_id: str | None = None


class SpeakerRegistration(BaseModel):
    """Request to register a speaker for voice cloning."""

    speaker_id: str


class TranslationResponse(BaseModel):
    """Response from translation."""

    text: str
    source_lang: str
    target_lang: str
    speaker_id: str | None = None
    audio_url: str | None = None


class TranslationServer:
    """Main translation server."""

    def __init__(self, config: TranslatorConfig | None = None):
        self.config = config or TranslatorConfig()
        self.pipeline = TranslationPipeline(self.config)
        self.active_connections: list[WebSocket] = []

    async def startup(self) -> None:
        """Initialize server and load models."""
        logger.info("Starting translation server...")
        await self.pipeline.load()
        logger.info("Server ready")

    async def shutdown(self) -> None:
        """Cleanup on shutdown."""
        logger.info("Shutting down server...")
        await self.pipeline.unload()


# Global server instance
_server: TranslationServer | None = None


def get_server() -> TranslationServer:
    """Get the global server instance."""
    global _server
    if _server is None:
        _server = TranslationServer()
    return _server


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    server = get_server()
    await server.startup()
    yield
    await server.shutdown()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Zen Translator API",
        description="Real-time multimodal translation with voice cloning and lip sync",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "version": "0.1.0"}

    # Translation endpoints
    @app.post("/translate/audio", response_model=TranslationResponse)
    async def translate_audio(
        audio: UploadFile = File(...),
        source_lang: str | None = Form(None),
        target_lang: str = Form("en"),
        speaker_id: str | None = Form(None),
    ):
        """Translate audio file."""
        server = get_server()

        # Read audio file
        audio_bytes = await audio.read()
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        result = await server.pipeline.translate_audio(
            audio=audio_array,
            source_lang=source_lang,
            target_lang=target_lang,
            speaker_id=speaker_id,
        )

        # Save audio to temp file if present
        audio_url = None
        if "audio" in result:
            import tempfile

            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, result["audio"], result["sample_rate"])
                audio_url = f"/audio/{Path(f.name).name}"

        return TranslationResponse(
            text=result["text"],
            source_lang=result["source_lang"],
            target_lang=result["target_lang"],
            speaker_id=result.get("speaker_id"),
            audio_url=audio_url,
        )

    @app.post("/translate/video")
    async def translate_video(
        video: UploadFile = File(...),
        source_lang: str | None = Form(None),
        target_lang: str = Form("en"),
        speaker_id: str | None = Form(None),
    ):
        """Translate video with lip sync."""
        server = get_server()

        # Save uploaded video
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = Path(f.name)
            f.write(await video.read())

        output_path = video_path.parent / f"{video_path.stem}_translated.mp4"

        result = await server.pipeline.translate_video(
            video=video_path,
            source_lang=source_lang,
            target_lang=target_lang,
            speaker_id=speaker_id,
            output_path=output_path,
        )

        # Cleanup input
        video_path.unlink()

        return FileResponse(
            result["output_path"],
            media_type="video/mp4",
            filename="translated_video.mp4",
        )

    @app.post("/speakers/register")
    async def register_speaker(
        speaker_id: str = Form(...),
        audio: UploadFile = File(...),
    ):
        """Register a speaker for voice cloning."""
        server = get_server()

        # Read audio
        audio_bytes = await audio.read()
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        result = await server.pipeline.register_speaker(
            speaker_id=speaker_id,
            reference_audio=audio_array,
        )

        return result

    @app.get("/speakers")
    async def list_speakers():
        """List registered speakers."""
        server = get_server()
        return {"speakers": server.pipeline.voice_cloner.list_speakers()}

    @app.get("/languages")
    async def get_languages():
        """Get supported languages."""
        server = get_server()
        return server.pipeline.get_supported_languages()

    # WebSocket for real-time streaming
    @app.websocket("/ws/translate")
    async def websocket_translate(websocket: WebSocket):
        """WebSocket endpoint for real-time translation."""
        server = get_server()
        await websocket.accept()
        server.active_connections.append(websocket)

        try:
            # Receive configuration
            config_data = await websocket.receive_json()
            source_lang = config_data.get("source_lang")
            target_lang = config_data.get("target_lang", "en")
            speaker_id = config_data.get("speaker_id")

            await websocket.send_json({"status": "ready", "message": "Send audio chunks"})

            # Create audio stream
            async def audio_generator():
                while True:
                    try:
                        data = await websocket.receive_bytes()
                        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                        yield audio
                    except WebSocketDisconnect:
                        break

            # Stream translation
            async for result in server.pipeline.stream_translate(
                audio_stream=audio_generator(),
                source_lang=source_lang,
                target_lang=target_lang,
                speaker_id=speaker_id,
            ):
                # Send text
                await websocket.send_json(
                    {
                        "type": "text",
                        "text": result["text"],
                    }
                )

                # Send audio
                if "audio" in result:
                    audio_bytes = (result["audio"] * 32768).astype(np.int16).tobytes()
                    await websocket.send_bytes(audio_bytes)

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        finally:
            server.active_connections.remove(websocket)

    return app


# CLI entry point
def main():
    """Run the translation server."""
    import uvicorn

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
