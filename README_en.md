# local-ASR-TTS-LLM-realtime

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

High-performance real-time voice assistant with client-server architecture. Resource-intensive AI models run on local server, while client operates on low-power embedded devices (e.g., Orange Pi), achieving fully private, low-latency voice interactions.

[English Version](./README_en.md) | [中文版本](./README.md)

## Core Features

* **Low-Latency Interaction**: TTFA (Time To First Audio) under 2 seconds from speech end to AI response start
* **Streaming Response**: LLM and TTS streaming processing drastically reduces waiting time
* **Fully Local**: All models run locally, ensuring complete data privacy
* **Precise Performance Monitoring**: Detailed metrics for accurate latency analysis
* **Highly Tunable VAD**: Multi-level Voice Activity Detection parameters for various environments

## Tech Stack

- **ASR**: SenseVoice (FunASR) - High-accuracy Chinese/English speech recognition
- **LLM**: Ollama + Qwen2.5:7B FP8 - Streaming conversation generation
- **TTS**: index-tts-vllm - High-quality speech synthesis with voice cloning
- **Client**: Low-power embedded devices (Orange Pi Zero3, etc.)
- **Server**: PC + NVIDIA GPU

## Project Structure

```
├── realtime_voice_assistant.py    # Client main program
├── asr_sensevoice.py              # ASR server
├── wavs/                          # TTS reference audio files
└── requirements.txt               # Dependencies
```

**Core Modules**:
- `ASRClient`: Audio recording, VAD, ASR communication
- `LLMClient`: Conversation history management, Ollama streaming
- `TTSProcessor`: Intelligent text segmentation, concurrent TTS requests
- `AudioPlayer`: Independent audio playback thread
- `VoiceAssistant`: Main controller orchestrating all modules
- `ConversationMetrics`: Performance timestamp analysis

## Server Deployment

### 1. ASR Service (SenseVoice)

Using the provided `asr_sensevoice.py`:

```bash
# Install dependencies
pip install fastapi uvicorn funasr torch
# For CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Start ASR service
python asr_sensevoice.py --port 8001 --model-path ./asr_models
```

Models download automatically on first run.

### 2. TTS Service (index-tts-vllm)

Refer to [Ksuriuri/index-tts-vllm](https://github.com/Ksuriuri/index-tts-vllm) for deployment:

```bash
# Typical startup command (refer to project docs for specifics)
python -m vllm.entrypoints.openai.api_server \
    --model path/to/index-tts-model \
    --port 11996
```

### 3. LLM Service (Ollama)

```bash
# Install Ollama then pull model
ollama pull qwen2.5:7b-fp8

# Start service (Important: bind to LAN IP)
$env:OLLAMA_HOST="xxx.xxx.x.xxx"; ollama serve  # windows
```

## Client Usage

### Requirements
- Python 3.10+
- alsa-utils (Linux audio support)
- Low-power single-board computer (Orange Pi, Raspberry Pi, etc.)

### Install Dependencies

```bash
# System dependencies
sudo apt-get update && sudo apt-get install alsa-utils

# Python dependencies
pip3 install numpy ollama pygame-ce requests
```

### Start Assistant

```bash
python3 realtime_voice_assistant.py \
  --asr-url http://<SERVER_IP>:8001 \
  --tts-url http://<SERVER_IP>:11996/tts_url \
  --ollama-host <SERVER_IP>:11434 \
  --device 3 \
  --volume-threshold 1.0 \
  --silence-duration 0.8   # It seems this parameter doesn’t work—in practice, hardcoded parameters are being used.
```

**Key Parameters**:
- `--device`: Recording device ID (check with `arecord -l`)
- `--volume-threshold`: Volume change threshold (dB) to trigger recording, lower = more sensitive
- `--silence-duration`: Silence duration (seconds) to detect speech end

## Common Issues & Solutions

### 1. Cannot Connect to Ollama
**Symptom**: Client connection errors  
**Solution**: Ensure Ollama binds to `0.0.0.0`, not default `127.0.0.1`

```bash
OLLAMA_HOST=0.0.0.0 ollama serve
```

### 2. No Audio Output
**Symptom**: Program runs but no speaker output  
**Solution**: Linux SBC Pygame driver issue, force alsa at script beginning

```python
import os
os.environ['SDL_AUDIODRIVER'] = 'alsa'
import pygame
```

### 3. Sluggish VAD Response
**Symptom**: Long delay after finishing speech  
**Quick Fix**: Lower `--silence-duration` to 0.6 or smaller  
**Fine Tuning**: Modify `dynamic_silence_threshold` and `effective_threshold` in `ASRClient.start_listening`

### 4. ASR Model Loading Failure
**Symptom**: SenseVoice service startup errors  
**Solutions**: 
- Check network, first run downloads models
- Ensure sufficient disk space
- GPU memory insufficient will auto-fallback to CPU

### 5. Poor TTS Quality
**Symptom**: Unnatural synthesized speech  
**Solutions**: 
- Prepare high-quality reference audio (16kHz, wav format)
- Adjust index-tts-vllm inference parameters
- Ensure reference audio duration 10-30 seconds

### 6. Command-line Parameters Ineffective
**Symptom**: Parameter changes don't affect behavior  
**Root Solution**: Directly modify default values in corresponding class `__init__` methods

## Customization & Extension

### Change LLM Model
```python
# In LLMClient.__init__ modify
self.model = "qwen2.5:7b-instruct-q8_0"  # or other ollama models
```

### Change TTS Voice
```python
# In VoiceAssistant._process_conversation modify
self.tts_processor = TTSProcessor(
    audio_paths=["wavs/custom_voice.wav"],  # your reference audio
    # ...
)
```

### Adapt Other Services
- **ASR**: Rewrite `ASRClient.send_to_asr` method
- **TTS**: Rewrite `TTSProcessor.generate_tts_audio` method
- **LLM**: Rewrite `LLMClient` streaming interaction logic

### Performance Optimization Tips
1. **ASR**: Adjust batch_size and VAD parameters
2. **TTS**: Enable vllm tensor parallelism
3. **Network**: Use gigabit network to reduce transmission latency
4. **Hardware**: High-performance GPU for server, low-latency audio devices for client

## System Architecture

```
[Microphone] -> [Client] --(Audio)--> [ASR:8001] -> [LLM:11434] -> [TTS:11996] -> [Client] -> [Speaker]
               Orange Pi                SenseVoice    Qwen2.5      index-tts    Audio Stream
```

---
1.Contact: cialtion@outlook.com | cialtion737410@sjtu.edu.cn

2.Notes: https://haxxorcialtion.github.io/sub_htmls/局域网内香橙派完全本地ASR-LLM-TTS_Realtime架构.html
