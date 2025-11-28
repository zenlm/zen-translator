"""Zen Translator WebRTC UI.

Real-time translation with voice cloning and lip synchronization.
Uses local Qwen3-Omni + CosyVoice + Wav2Lip pipeline instead of cloud APIs.
"""

import asyncio
import signal
from pathlib import Path

import gradio as gr
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

try:
    from fastrtc import (
        AdditionalOutputs,
        AsyncAudioVideoStreamHandler,
        Stream,
        get_cloudflare_turn_credentials_async,
        wait_for_item,
    )

    FASTRTC_AVAILABLE = True
except ImportError:
    FASTRTC_AVAILABLE = False
    AdditionalOutputs = None
    AsyncAudioVideoStreamHandler = object
    Stream = None

from gradio.utils import get_space

from ..config import TranslatorConfig
from ..pipeline import TranslationPipeline

cur_dir = Path(__file__).parent

# Language mappings
LANG_MAP = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "tr": "Turkish",
    "pl": "Polish",
    # Chinese dialects
    "yue": "Cantonese",
    "wuu": "Shanghainese",
    "hsn": "Xiang",
    "nan": "Min Nan",
    "hak": "Hakka",
    "cdo": "Min Dong",
}
LANG_MAP_REVERSE = {v: k for k, v in LANG_MAP.items()}

# Input languages (18 + 6 dialects)
INPUT_LANGUAGES = [
    LANG_MAP[code]
    for code in [
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
        "yue",
        "wuu",
        "hsn",
        "nan",
        "hak",
        "cdo",
    ]
]

# Output languages (10)
OUTPUT_LANGUAGES = [
    LANG_MAP[code] for code in ["en", "zh", "ja", "ko", "es", "fr", "de", "it", "pt", "ru"]
]

# Default voices (from CosyVoice)
VOICES = ["default", "male_1", "female_1", "male_2", "female_2"]

# Global pipeline instance
_pipeline: TranslationPipeline | None = None
_pipeline_lock = asyncio.Lock()


async def get_pipeline() -> TranslationPipeline:
    """Get or initialize the translation pipeline."""
    global _pipeline
    async with _pipeline_lock:
        if _pipeline is None:
            config = TranslatorConfig()
            _pipeline = TranslationPipeline(config)
            await _pipeline.load()
        return _pipeline


class ZenTranslateHandler(AsyncAudioVideoStreamHandler):
    """Local translation handler using Zen Translator pipeline."""

    def __init__(self) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=24_000,
            input_sample_rate=16_000,
        )
        self.output_queue: asyncio.Queue = asyncio.Queue()
        self.video_queue: asyncio.Queue = asyncio.Queue()

        self.last_send_time = 0.0
        self.video_interval = 0.5  # Send video frames every 0.5s
        self.latest_frame = None

        self.awaiting_new_message = True
        self.stable_text = ""
        self.temp_text = ""

        # Audio buffer for streaming
        self.audio_buffer: list[np.ndarray] = []
        self.buffer_duration_ms = 500  # Process every 500ms of audio

        self.pipeline: TranslationPipeline | None = None
        self.source_lang = "en"
        self.target_lang = "zh"
        self.speaker_id: str | None = None
        self._running = False

    def copy(self):
        return ZenTranslateHandler()

    async def start_up(self):
        """Initialize the translation pipeline and start processing."""
        try:
            await self.wait_for_args()
            args = self.latest_args

            # Parse arguments from UI
            src_language_name = args[2] if len(args) > 2 else "English"
            target_language_name = args[3] if len(args) > 3 else "Chinese"
            self.source_lang = LANG_MAP_REVERSE.get(src_language_name, "en")
            self.target_lang = LANG_MAP_REVERSE.get(target_language_name, "zh")

            voice_id = args[4] if len(args) > 4 else "default"
            if voice_id != "default":
                self.speaker_id = voice_id

            # Get pipeline
            self.pipeline = await get_pipeline()
            self._running = True

            # Process audio buffer continuously
            await self._process_loop()

        except Exception as e:
            print(f"Translation error: {e}")
            await self.shutdown()

    async def _process_loop(self):
        """Main processing loop for buffered audio."""
        sample_rate = 16000
        samples_per_buffer = int(sample_rate * self.buffer_duration_ms / 1000)

        while self._running:
            # Wait for enough audio data
            total_samples = sum(len(chunk) for chunk in self.audio_buffer)

            if total_samples >= samples_per_buffer:
                # Combine audio chunks
                combined = np.concatenate(self.audio_buffer)
                self.audio_buffer = []

                # Translate
                try:
                    result = await self.pipeline.translate_audio(
                        audio=combined,
                        source_lang=self.source_lang,
                        target_lang=self.target_lang,
                        speaker_id=self.speaker_id,
                    )

                    # Send text update
                    text = result.get("text", "")
                    if text:
                        self.stable_text = text
                        await self.output_queue.put(
                            AdditionalOutputs(
                                {
                                    "role": "assistant",
                                    "content": f"<span style='color:black'>{self.stable_text}</span>",
                                    "update": True,
                                    "new_message": self.awaiting_new_message,
                                }
                            )
                        )
                        self.awaiting_new_message = False

                    # Send audio output
                    audio = result.get("audio")
                    if audio is not None:
                        audio_out = audio.astype(np.int16).reshape(1, -1)
                        await self.output_queue.put((self.output_sample_rate, audio_out))

                except Exception as e:
                    print(f"Translation chunk error: {e}")

            else:
                await asyncio.sleep(0.05)

    async def video_receive(self, frame: np.ndarray):
        """Receive video frame from client."""
        self.latest_frame = frame
        await self.video_queue.put(frame)

    async def video_emit(self):
        """Emit video frame to client."""
        if self.latest_frame is not None:
            return self.latest_frame
        return np.zeros((480, 640, 3), dtype=np.uint8)

    async def receive(self, frame):
        """Receive audio frame from client."""
        if not self._running:
            return
        sr, array = frame
        array = array.squeeze()

        # Add to buffer
        self.audio_buffer.append(array)

    async def emit(self) -> tuple[int, np.ndarray] | AdditionalOutputs | None:
        """Emit audio/text output to client."""
        return await wait_for_item(self.output_queue)

    async def shutdown(self) -> None:
        """Clean up resources."""
        self._running = False

        # Clear queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break


def update_chatbot(chatbot: list[dict], response: dict):
    """Update chatbot with new message."""
    is_update = response.pop("update", False)
    new_message_flag = response.pop("new_message", False)
    content = response["content"]

    if is_update:
        if new_message_flag or not chatbot:
            chatbot.append({"role": "assistant", "content": content})
        else:
            if chatbot and chatbot[-1]["role"] == "assistant":
                chatbot[-1]["content"] = content
            else:
                chatbot.append({"role": "assistant", "content": content})
    else:
        chatbot.append(response)

    return chatbot


def create_demo() -> gr.Blocks:
    """Create Gradio demo without FastRTC (fallback mode)."""
    with gr.Blocks(title="Zen Translator") as demo:
        gr.Markdown(
            """
        # 🎙️ Zen Translator

        Real-time multimodal translation with voice cloning and lip synchronization.

        **Note**: This is the file-based interface. For real-time WebRTC streaming,
        install fastrtc and run with `zen-serve --mode webrtc`.
        """
        )

        with gr.Row():
            with gr.Column():
                src_lang = gr.Dropdown(
                    choices=INPUT_LANGUAGES,
                    value="English",
                    label="Source Language",
                )
                target_lang = gr.Dropdown(
                    choices=OUTPUT_LANGUAGES,
                    value="Chinese",
                    label="Target Language",
                )
                voice = gr.Dropdown(
                    choices=VOICES,
                    value="default",
                    label="Voice",
                )

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    label="Input Audio",
                    type="numpy",
                    sources=["microphone", "upload"],
                )
                video_input = gr.Video(
                    label="Input Video (optional)",
                    sources=["webcam", "upload"],
                )

            with gr.Column():
                audio_output = gr.Audio(
                    label="Translated Audio",
                    type="numpy",
                )
                video_output = gr.Video(
                    label="Output Video (with lip sync)",
                )
                text_output = gr.Textbox(
                    label="Translation",
                    lines=3,
                )

        translate_btn = gr.Button("Translate", variant="primary")

        async def translate_file(audio, video, src, target, voice_id):
            """Translate uploaded file."""
            pipeline = await get_pipeline()

            src_code = LANG_MAP_REVERSE.get(src, "en")
            target_code = LANG_MAP_REVERSE.get(target, "zh")

            if video is not None:
                # Video translation with lip sync
                result = await pipeline.translate_video(
                    video=video,
                    source_lang=src_code,
                    target_lang=target_code,
                    speaker_id=voice_id if voice_id != "default" else None,
                )
                return (
                    (result.get("sample_rate", 24000), result.get("audio"))
                    if result.get("audio") is not None
                    else None,
                    result.get("output_path"),
                    result.get("text", ""),
                )
            elif audio is not None:
                # Audio-only translation
                sr, audio_data = audio
                result = await pipeline.translate_audio(
                    audio=audio_data,
                    source_lang=src_code,
                    target_lang=target_code,
                    speaker_id=voice_id if voice_id != "default" else None,
                )
                return (
                    (result.get("sample_rate", 24000), result.get("audio"))
                    if result.get("audio") is not None
                    else None,
                    None,
                    result.get("text", ""),
                )
            else:
                return None, None, "Please provide audio or video input."

        translate_btn.click(
            fn=translate_file,
            inputs=[audio_input, video_input, src_lang, target_lang, voice],
            outputs=[audio_output, video_output, text_output],
        )

    return demo


def create_stream() -> "Stream":
    """Create FastRTC stream for WebRTC mode."""
    if not FASTRTC_AVAILABLE:
        raise RuntimeError("fastrtc not installed. Run: pip install fastrtc")

    chatbot = gr.Chatbot(type="messages")
    src_language = gr.Dropdown(
        choices=INPUT_LANGUAGES,
        value="English",
        type="value",
        label="Source Language",
    )
    target_language = gr.Dropdown(
        choices=OUTPUT_LANGUAGES,
        value="Chinese",
        type="value",
        label="Target Language",
    )
    voice = gr.Dropdown(
        choices=VOICES,
        value="default",
        type="value",
        label="Voice",
    )

    rtc_config = get_cloudflare_turn_credentials_async if get_space() else None

    stream = Stream(
        ZenTranslateHandler(),
        mode="send-receive",
        modality="audio-video",
        additional_inputs=[src_language, target_language, voice, chatbot],
        additional_outputs=[chatbot],
        additional_outputs_handler=update_chatbot,
        rtc_configuration=rtc_config,
        concurrency_limit=5 if get_space() else None,
        time_limit=90 if get_space() else None,
    )

    return stream


def create_app() -> FastAPI:
    """Create FastAPI app with WebRTC support."""
    if not FASTRTC_AVAILABLE:
        # Fallback to simple Gradio app
        demo = create_demo()
        app = FastAPI()
        app = gr.mount_gradio_app(app, demo, path="/")
        return app

    stream = create_stream()
    app = FastAPI()
    stream.mount(app)

    @app.get("/")
    async def index():
        rtc_config = await get_cloudflare_turn_credentials_async() if get_space() else None
        html_path = cur_dir / "index.html"
        if html_path.exists():
            html_content = html_path.read_text()
            import json

            html_content = html_content.replace("__RTC_CONFIGURATION__", json.dumps(rtc_config))
            return HTMLResponse(content=html_content)
        else:
            # Use default stream UI
            return HTMLResponse(
                content="""
                <html>
                <head><title>Zen Translator</title></head>
                <body>
                <h1>Zen Translator</h1>
                <p>WebRTC streaming mode. Visit /ui for Gradio interface.</p>
                </body>
                </html>
            """
            )

    @app.get("/outputs")
    def outputs(webrtc_id: str):
        import json

        async def output_stream():
            async for output in stream.output_stream(webrtc_id):
                s = json.dumps(output.args[0])
                yield f"event: output\ndata: {s}\n\n"

        return StreamingResponse(output_stream(), media_type="text/event-stream")

    # Mount Gradio UI at /ui
    demo = create_demo()
    app = gr.mount_gradio_app(app, demo, path="/ui")

    return app


def handle_exit(sig, frame):
    """Handle graceful shutdown."""
    print("\nShutting down Zen Translator...")
    exit(0)


signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

if __name__ == "__main__":
    import os

    import uvicorn

    mode = os.getenv("MODE", "UI")

    if mode == "UI":
        demo = create_demo()
        demo.launch(server_port=7860)
    elif mode == "WEBRTC":
        if FASTRTC_AVAILABLE:
            stream = create_stream()
            stream.ui.launch(server_port=7860)
        else:
            print("fastrtc not available, falling back to UI mode")
            demo = create_demo()
            demo.launch(server_port=7860)
    else:
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=8000)
