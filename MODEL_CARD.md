---
license: apache-2.0
language:
  - en
  - zh
  - ja
  - ko
  - es
  - fr
  - de
  - it
  - pt
  - ru
library_name: transformers
pipeline_tag: audio-to-audio
tags:
  - translation
  - voice-cloning
  - lip-sync
  - multimodal
  - real-time
  - qwen3-omni
  - cosyvoice
  - wav2lip
  - hanzo-ai
  - zen-lm
---

# Zen Translator

Real-time multimodal translation with voice cloning and lip synchronization.

## Overview

Zen Translator combines three state-of-the-art models into a sub-second end-to-end pipeline:

| Component | Model | Parameters | Latency |
|-----------|-------|------------|---------|
| Translation | [Qwen3-Omni-30B-A3B](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | 30B (3B active MoE) | ~500ms |
| Voice Cloning | [CosyVoice 2.0](https://github.com/FunAudioLLM/CosyVoice) | 0.5B | ~150ms |
| Lip Sync | [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) | ~100M | ~200ms |
| **Total** | - | - | **<1 second** |

## Features

- **18 input languages** including Chinese dialects (Cantonese, Shanghainese, etc.)
- **10 output languages** with high-quality voice synthesis
- **3-second voice cloning** - Preserve speaker characteristics with minimal reference audio
- **Real-time streaming** - WebSocket API with <500ms first packet latency
- **Lip synchronization** - Natural video dubbing for translated content
- **News anchor training** - Domain-specific finetuning for broadcast translation

## Quick Start

```bash
# Clone repository
git clone https://github.com/zenlm/zen-translator.git
cd zen-translator

# Install with uv
make install

# Download models (~62GB full, ~16GB quantized)
make download
# OR
make download-quantized

# Start server
make serve
```

## Usage

### Python API

```python
from zen_translator import TranslationPipeline, TranslatorConfig

config = TranslatorConfig(target_language="es")
pipeline = TranslationPipeline(config)
await pipeline.load()

# Register speaker voice (3+ seconds of audio)
await pipeline.register_speaker("john_doe", "reference.wav")

# Translate video with voice cloning and lip sync
result = await pipeline.translate_video(
    video="news.mp4",
    target_lang="es",
    speaker_id="john_doe",
    output_path="news_es.mp4"
)
```

### CLI

```bash
# Translate a video
zen-translate video.mp4 -o translated.mp4 -t spanish

# Register a speaker
zen-translate register-speaker john_doe reference.wav

# Start the API server
zen-serve --host 0.0.0.0 --port 8000
```

### REST API

```bash
# Translate audio
curl -X POST http://localhost:8000/translate/audio \
  -F "audio=@input.wav" \
  -F "target_lang=es"

# Translate video with lip sync
curl -X POST http://localhost:8000/translate/video \
  -F "video=@input.mp4" \
  -F "target_lang=zh"
```

### WebSocket (Real-time)

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/translate');
ws.send(JSON.stringify({ target_lang: 'es', speaker_id: 'my_voice' }));
ws.send(audioChunk);  // Send audio chunks
ws.onmessage = (event) => {
    // Receive translated audio chunks
};
```

## Language Support

### Input Languages (18 + 6 dialects)

| Language | Code |
|----------|------|
| English | en |
| Chinese | zh |
| Japanese | ja |
| Korean | ko |
| Spanish | es |
| French | fr |
| German | de |
| Italian | it |
| Portuguese | pt |
| Russian | ru |
| Arabic | ar |
| Hindi | hi |
| Thai | th |
| Vietnamese | vi |
| Indonesian | id |
| Malay | ms |
| Turkish | tr |
| Polish | pl |
| **Dialects** | |
| Cantonese | yue |
| Shanghainese | wuu |
| Xiang | hsn |
| Min Nan | nan |
| Hakka | hak |
| Min Dong | cdo |

### Output Languages (10)

English, Chinese, Japanese, Korean, Spanish, French, German, Italian, Portuguese, Russian

## Model Requirements

| Model | VRAM | Disk |
|-------|------|------|
| Qwen3-Omni | 16GB | 60GB |
| CosyVoice 2.0 | 2GB | 1GB |
| Wav2Lip | 2GB | 500MB |
| **Total** | **~20GB** | **~62GB** |

For smaller deployments, use 4-bit quantized Qwen3-Omni (~15GB disk).

## Training

### News Anchor Adaptation

```bash
# Build dataset from news channels (CNN, BBC, NHK, DW)
make dataset-build

# Train news anchor adaptation
make train-anchor

# Or with ms-swift directly
swift sft --config outputs/anchor/train_config.yaml
```

## Citation

```bibtex
@software{zen_translator,
  author = {Hanzo AI and Zen LM},
  title = {Zen Translator: Real-time Multimodal Translation with Voice Cloning},
  year = {2025},
  url = {https://github.com/zenlm/zen-translator}
}
```

## Links

- **GitHub**: https://github.com/zenlm/zen-translator
- **Zen LM**: https://zenlm.org
- **Hanzo AI**: https://hanzo.ai

## License

Apache 2.0
