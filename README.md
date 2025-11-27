# Zen Translator

Real-time multimodal translation with voice cloning and lip synchronization.

Built on:
- **[Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)** - Real-time speech understanding and translation
- **[CosyVoice 2.0](https://github.com/FunAudioLLM/CosyVoice)** - Ultra-low latency voice cloning (150ms)
- **[Wav2Lip](https://github.com/Rudrabha/Wav2Lip)** - Accurate lip synchronization

## Features

- 🌐 **18 input languages**, 10 output languages
- 🎙️ **3-second voice cloning** - Preserve speaker characteristics
- 👄 **Accurate lip sync** - Natural video dubbing
- ⚡ **<1 second latency** - Real-time streaming
- 📺 **News anchor optimization** - Domain-specific finetuning

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/zenlm/zen-translator.git
cd zen-translator

# Install with uv
make install

# Download models (requires ~100GB disk space)
make download
```

### Usage

**Translate a video:**
```bash
zen-translate video.mp4 -o translated.mp4 -t spanish
```

**Start the API server:**
```bash
make serve
# Server runs at http://localhost:8000
```

**Real-time WebSocket translation:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/translate');
ws.send(JSON.stringify({ target_lang: 'es', speaker_id: 'my_voice' }));
ws.send(audioChunk);  // Send audio chunks
ws.onmessage = (event) => {
    // Receive translated audio chunks
};
```

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

## Supported Languages

### Input (18 + 6 dialects)
English, Chinese, Japanese, Korean, Spanish, French, German, Italian, Portuguese, Russian, Arabic, Hindi, Thai, Vietnamese, Indonesian, Malay, Turkish, Polish, Cantonese, Shanghainese, and more.

### Output (10)
English, Chinese, Japanese, Korean, Spanish, French, German, Italian, Portuguese, Russian

## Voice Cloning

Register a speaker with just 3 seconds of audio:

```python
from zen_translator import TranslationPipeline

pipeline = TranslationPipeline()
await pipeline.load()

# Register speaker
await pipeline.register_speaker(
    speaker_id="john_doe",
    reference_audio="reference.wav"
)

# Translate with cloned voice
result = await pipeline.translate_audio(
    audio="input.wav",
    target_lang="es",
    speaker_id="john_doe"
)
```

## News Anchor Training

Finetune for accurate news translation:

```bash
# Build dataset from news channels
make dataset-build

# Train news anchor adaptation
make train-anchor

# Or with ms-swift directly
swift sft --config outputs/anchor/train_config.yaml
```

Supported news sources:
- CNN, BBC News, NHK World, DW News
- France24, Al Jazeera, Sky News, Reuters
- CCTV, TBS, KBS, and more

## API Reference

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/translate/audio` | POST | Translate audio file |
| `/translate/video` | POST | Translate video with lip sync |
| `/speakers/register` | POST | Register voice for cloning |
| `/speakers` | GET | List registered speakers |
| `/languages` | GET | Get supported languages |
| `/ws/translate` | WS | Real-time streaming translation |

### Python API

```python
from zen_translator import TranslationPipeline, TranslatorConfig

# Configure
config = TranslatorConfig(
    target_language="es",
    enable_lip_sync=True,
    preserve_emotion=True,
)

# Initialize
pipeline = TranslationPipeline(config)
await pipeline.load()

# Translate video
result = await pipeline.translate_video(
    video="news_clip.mp4",
    output_path="translated.mp4",
)
```

## Model Requirements

| Model | Parameters | VRAM | Disk |
|-------|------------|------|------|
| Qwen3-Omni | 30B (3B active) | 16GB | 60GB |
| CosyVoice 2.0 | 0.5B | 2GB | 1GB |
| Wav2Lip | ~100M | 2GB | 500MB |
| **Total** | - | **~20GB** | **~62GB** |

For smaller deployments, use quantized models:
```bash
make download-quantized  # 4-bit Qwen3-Omni (~15GB)
```

## Development

```bash
# Install dev dependencies
make dev

# Run tests
make test

# Lint and format
make lint format

# Type check
make typecheck
```

## Configuration

Environment variables:
```bash
export ZEN_TRANSLATOR_TARGET_LANGUAGE=es
export ZEN_TRANSLATOR_DEVICE=cuda
export ZEN_TRANSLATOR_DTYPE=bfloat16
export ZEN_TRANSLATOR_ENABLE_LIP_SYNC=true
```

Or use `.env` file in project root.

## License

Apache 2.0

## Credits

- **Qwen Team** - Qwen3-Omni model
- **Alibaba FunAudioLLM** - CosyVoice
- **Wav2Lip Authors** - Lip synchronization
- **Hanzo AI / Zen LM** - Integration and finetuning

## Links

- [Zen LM](https://zenlm.org)
- [Qwen3-Omni](https://huggingface.co/collections/Qwen/qwen3-omni)
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- [Wav2Lip](https://github.com/Rudrabha/Wav2Lip)
