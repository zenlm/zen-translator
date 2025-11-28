# Zen Translator - AI Knowledge Base

**Project**: zen-translator
**Organization**: zenlm
**Repository**: https://github.com/zenlm/zen-translator
**Version**: 0.1.0
**Last Updated**: 2025-11-28

## Project Overview

Zen Translator is a real-time multimodal translation pipeline that combines speech translation, voice cloning, and lip synchronization for seamless video dubbing and live translation.

### Core Technology Stack

| Component | Model | Parameters | Latency |
|-----------|-------|------------|---------|
| Translation | Qwen3-Omni-30B-A3B | 30B (3B active MoE) | ~500ms |
| Voice Cloning | CosyVoice 2.0 | 0.5B | ~150ms |
| Lip Sync | Wav2Lip | ~100M | ~200ms |
| **Total** | - | - | **<1 second** |

### Language Support

**Input (18 languages + 6 dialects)**:
- English, Chinese, Japanese, Korean, Spanish, French, German, Italian, Portuguese, Russian
- Arabic, Hindi, Thai, Vietnamese, Indonesian, Malay, Turkish, Polish
- Cantonese (yue), Shanghainese (wuu), Xiang (hsn), Min Nan (nan), Hakka (hak), Min Dong (cdo)

**Output (10 languages)**:
- English, Chinese, Japanese, Korean, Spanish, French, German, Italian, Portuguese, Russian

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Zen Translator Pipeline                     │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  Audio/Video    │  Qwen3-Omni     │  Translation + Understanding │
│  Input          │  (30B MoE)      │  ~500ms                       │
├─────────────────┼─────────────────┼─────────────────────────────┤
│  Translated     │  CosyVoice 2.0  │  Voice Cloning               │
│  Text           │  (0.5B)         │  ~150ms                       │
├─────────────────┼─────────────────┼─────────────────────────────┤
│  Cloned Audio   │  Wav2Lip        │  Lip Synchronization         │
│  + Video        │                 │  ~200ms                       │
├─────────────────┴─────────────────┴─────────────────────────────┤
│  Total End-to-End Latency: <1 second                            │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
zen-translator/
├── src/zen_translator/
│   ├── __init__.py           # Package exports
│   ├── config.py             # TranslatorConfig, NewsAnchorConfig
│   ├── pipeline.py           # Main TranslationPipeline orchestrator
│   ├── cli.py                # Typer CLI (zen-translate command)
│   ├── translation/
│   │   ├── __init__.py
│   │   └── qwen3_omni.py     # Qwen3-Omni translation
│   ├── voice_clone/
│   │   ├── __init__.py
│   │   └── cosyvoice.py      # CosyVoice 2.0 voice cloning
│   ├── lip_sync/
│   │   ├── __init__.py
│   │   ├── wav2lip.py        # Wav2Lip lip synchronization
│   │   └── wav2lip_model.py  # Wav2Lip neural network architecture
│   ├── streaming/
│   │   ├── __init__.py
│   │   └── server.py         # FastAPI + WebSocket server
│   ├── ui/
│   │   ├── __init__.py       # UI module exports
│   │   └── app.py            # Gradio + FastRTC WebRTC UI
│   ├── server.py             # Standalone server entry point
│   └── training/
│       ├── __init__.py
│       ├── swift_config.py       # ms-swift finetuning configs
│       └── news_anchor_dataset.py # News anchor data collection
├── configs/
│   ├── train_identity.yaml   # Zen identity finetuning
│   └── train_anchor.yaml     # News anchor adaptation
├── scripts/
│   └── download_models.py    # Model download utility
├── tests/                    # Test suite
├── data/                     # Training data directory
│   ├── news_anchors/
│   └── voices/
├── models/                   # Downloaded model cache
├── pyproject.toml            # Package configuration (uv/pip)
├── Makefile                  # Build automation
├── README.md                 # User documentation
└── LLM.md                    # AI assistant knowledge base (this file)
```

## Key Components

### 1. TranslationPipeline (pipeline.py)

Main orchestrator that coordinates all translation stages:

```python
from zen_translator import TranslationPipeline, TranslatorConfig

config = TranslatorConfig(target_language="es")
pipeline = TranslationPipeline(config)
await pipeline.load()

# Audio translation
result = await pipeline.translate_audio(
    audio="input.wav",
    target_lang="es",
    speaker_id="john_doe"
)

# Video translation with lip sync
result = await pipeline.translate_video(
    video="news.mp4",
    target_lang="zh",
    output_path="news_zh.mp4"
)
```

### 2. Qwen3OmniTranslator (translation/qwen3_omni.py)

Handles speech understanding and translation using Qwen3-Omni:
- Audio input processing
- Video multimodal analysis (lip reading, visual context)
- Streaming translation support
- Built-in TTS when voice cloning not needed

### 3. CosyVoiceCloner (voice_clone/cosyvoice.py)

Voice cloning with 3-second reference audio:
- Speaker embedding extraction
- Emotion preservation
- Streaming synthesis (~150ms first packet)
- NewsAnchorVoiceBank for pre-registered voices

### 4. Wav2LipSync (lip_sync/wav2lip.py)

Lip synchronization for video dubbing:
- Face detection (face_alignment or OpenCV fallback)
- Mel spectrogram audio processing
- Batch processing for efficiency
- Quality presets: fast, balanced, quality

### 5. Gradio UI (ui/app.py)

Web interface for interactive translation with WebRTC streaming:

```python
# Launch Gradio UI
zen-translate ui --port 7860

# Or via Python
from zen_translator.ui import create_demo
demo = create_demo()
demo.launch()
```

**Modes:**
- `ui`: File-based Gradio interface (microphone/upload → translate)
- `webrtc`: Real-time FastRTC WebRTC streaming
- `api`: REST/WebSocket API server

**Features:**
- Source/target language selection (24 input, 10 output languages)
- Voice selection for TTS output
- Audio input: microphone or file upload
- Video input: webcam or file upload
- Real-time translation with voice cloning
- Lip sync for video output

### 6. TranslationServer (streaming/server.py)

FastAPI server for real-time translation:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/translate/audio` | POST | Translate audio file |
| `/translate/video` | POST | Translate video with lip sync |
| `/speakers/register` | POST | Register voice for cloning |
| `/speakers` | GET | List registered speakers |
| `/languages` | GET | Get supported languages |
| `/ws/translate` | WS | Real-time streaming translation |

## Configuration

### TranslatorConfig

```python
config = TranslatorConfig(
    # Models
    qwen3_omni_model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    cosyvoice_model="FunAudioLLM/CosyVoice2-0.5B",
    wav2lip_model="numz/wav2lip_studio",
    
    # Translation
    target_language="en",
    
    # Voice cloning
    voice_reference_seconds=3.0,
    preserve_emotion=True,
    
    # Lip sync
    enable_lip_sync=True,
    lip_sync_quality="balanced",
    
    # Hardware
    device="cuda",
    dtype="bfloat16",
    use_flash_attention=True,
)
```

### Environment Variables

```bash
ZEN_TRANSLATOR_TARGET_LANGUAGE=es
ZEN_TRANSLATOR_DEVICE=cuda
ZEN_TRANSLATOR_DTYPE=bfloat16
ZEN_TRANSLATOR_ENABLE_LIP_SYNC=true
```

## Training Infrastructure

### Identity Finetuning (ZenIdentityConfig)

Finetunes Qwen3-Omni with Zen Translator identity:
- Professional translation persona
- Consistent behavior and responses
- Uses ms-swift for LoRA training

### News Anchor Adaptation (NewsAnchorConfig)

Specialized training for broadcast translation:
- Collects data from YouTube news channels (CNN, BBC, NHK, DW, etc.)
- Segments into training samples
- Creates translation pairs
- Exports in ms-swift format

### Training Commands

```bash
# Build news anchor dataset
make dataset-build

# Generate training config
make train-anchor

# Run ms-swift training
swift sft --config outputs/anchor/train_config.yaml
```

## Development

### Setup

```bash
# Create venv and install
make install

# Install with dev dependencies
make dev

# Download models (~62GB full, ~16GB quantized)
make download
make download-quantized
```

### Testing

```bash
make test       # Run tests
make lint       # Run ruff linter
make format     # Format code
make typecheck  # Run mypy
```

### CLI Commands

```bash
# Translate file
zen-translate video.mp4 -o translated.mp4 -t spanish

# Start server
zen-serve --host 0.0.0.0 --port 8000

# Register speaker
zen-translate register-speaker john_doe reference.wav

# Download models
zen-translate download all

# Train
zen-translate train --type anchor --output ./outputs

# Launch UI
zen-translate ui --port 7860

# Launch with WebRTC
zen-translate ui --mode webrtc

# Launch API server
zen-serve --mode api --port 8000
```

## Model Requirements

| Model | Parameters | VRAM | Disk |
|-------|------------|------|------|
| Qwen3-Omni | 30B (3B active) | 16GB | 60GB |
| CosyVoice 2.0 | 0.5B | 2GB | 1GB |
| Wav2Lip | ~100M | 2GB | 500MB |
| **Total** | - | **~20GB** | **~62GB** |

For smaller deployments, use 4-bit quantized Qwen3-Omni (~15GB disk).

## Dependencies

### Core
- torch>=2.1.0
- transformers>=4.45.0
- accelerate>=0.25.0

### Audio
- librosa>=0.10.0
- soundfile>=0.12.0
- webrtcvad>=2.0.10

### Video
- opencv-python>=4.8.0
- ffmpeg-python>=0.2.0
- av>=11.0.0

### Streaming
- fastapi>=0.109.0
- uvicorn>=0.27.0
- websockets>=12.0

### Training
- ms-swift>=2.4.0
- peft>=0.7.0
- deepspeed>=0.13.0

### UI (optional)
- gradio>=4.44.0
- fastrtc>=0.0.20
- python-dotenv>=1.0.0

## Key Files

- `src/zen_translator/pipeline.py` - Main orchestration (line 23: TranslationPipeline)
- `src/zen_translator/translation/qwen3_omni.py` - Qwen3-Omni (line 28: Qwen3OmniTranslator)
- `src/zen_translator/voice_clone/cosyvoice.py` - CosyVoice (line 23: CosyVoiceCloner)
- `src/zen_translator/lip_sync/wav2lip.py` - Wav2Lip (line 21: Wav2LipSync)
- `src/zen_translator/streaming/server.py` - FastAPI server (line 92: create_app)
- `src/zen_translator/ui/app.py` - Gradio UI (line 117: ZenTranslateHandler, line 393: create_demo)
- `src/zen_translator/server.py` - Server entry point (line 13: main)

## Notes for AI Assistants

1. **ALWAYS** update this file with significant discoveries or changes
2. **NEVER** commit model files or weights (they're in .gitignore)
3. All Zen models are based on **Qwen3** (not Qwen2!)
4. Use `uv` for Python environment management
5. Use `make` commands for standard operations
6. The Wav2Lip model requires `wav2lip_model.py` for architecture definition
7. CosyVoice has fallback mode when not installed
8. Flash Attention 2 is recommended for performance

## Related Projects

- [zen](https://github.com/zenlm/zen) - Zen AI model family
- [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) - Base translation model
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - Voice cloning
- [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) - Lip synchronization
- [ms-swift](https://github.com/modelscope/ms-swift) - Training framework

---

**Zen Translator**: Real-time translation with voice cloning and lip sync.
