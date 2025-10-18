# 本地化实时语音助手 (local-ASR-TTS-LLM-realtime)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

基于客户端/服务器架构的高性能实时语音助手。将资源密集型AI模型部署在本地服务器，客户端运行在低功耗嵌入式设备（如香橙派），实现完全私有化、低延迟的语音交互体验。

[English Version](./README_en.md) | [中文版本](./README.md)

## 核心特性

* **低延迟交互**: 从用户说完话到AI第一个声音，TTFA控制在2秒以内
* **流式响应**: LLM和TTS流式处理，大幅缩短等待时间
* **完全本地化**: 所有模型本地运行，数据隐私保障
* **精确性能监控**: 详细性能指标统计，精确分析各环节耗时
* **高度可调VAD**: 多层级语音活动检测参数，适应不同环境和语速

## 技术栈

- **ASR**: SenseVoice (FunASR) - 高精度中英文语音识别
- **LLM**: Ollama + Qwen2.5:7B FP8 - 流式对话生成
- **TTS**: index-tts-vllm - 高质量语音合成，支持声音克隆
- **客户端**: 低功耗嵌入式设备（香橙派Zero3等）
- **服务端**: PC + NVIDIA GPU

## 项目结构

```
├── realtime_voice_assistant.py    # 客户端主程序
├── asr_sensevoice.py              # ASR服务端
├── wavs/                          # TTS参考音频文件目录
└── requirements.txt               # 依赖清单
```

**核心模块**:
- `ASRClient`: 音频录制、VAD、ASR通信
- `LLMClient`: 对话历史管理、Ollama流式交互  
- `TTSProcessor`: 文本智能分段、并发TTS请求
- `AudioPlayer`: 独立音频播放线程
- `VoiceAssistant`: 主控制器，调度所有模块
- `ConversationMetrics`: 性能时间戳分析

## 服务端部署

### 1. ASR服务 (SenseVoice)

使用提供的 `asr_sensevoice.py`:

```bash
# 安装依赖
pip install fastapi uvicorn funasr torch
# 如果需要CUDA支持
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 启动ASR服务
python asr_sensevoice.py --port 8001 --model-path ./asr_models
```

模型会自动下载到指定路径。首次启动较慢。

### 2. TTS服务 (index-tts-vllm)

参考 [Ksuriuri/index-tts-vllm](https://github.com/Ksuriuri/index-tts-vllm) 项目部署：

```bash
# 典型启动命令（具体参数参考该项目文档）
python -m vllm.entrypoints.openai.api_server \
    --model path/to/index-tts-model \
    --port 11996
```

### 3. LLM服务 (Ollama)

```bash
# 安装Ollama后拉取模型
ollama pull qwen2.5:7b-fp8

# 启动服务（重要：绑定局域网IP）
$env:OLLAMA_HOST="xxx.xxx.x.xxx"; ollama serve
```

## 客户端使用

### 系统要求
- Python 3.10+
- alsa-utils (Linux音频支持)
- 低功耗单板计算机（香橙派、树莓派等）

### 安装依赖

```bash
# 系统依赖
sudo apt-get update && sudo apt-get install alsa-utils

# Python依赖  
pip3 install numpy ollama pygame-ce requests
```

### 启动助手

```bash
python3 realtime_voice_assistant.py \
  --asr-url http://<服务器IP>:8001 \
  --tts-url http://<服务器IP>:11996/tts_url \
  --ollama-host <服务器IP>:11434 \
  --device 3 \
  --volume-threshold 1.0 \
  --silence-duration 0.8    # 这个参数貌似没用，实际使用了硬编码的参数
```

**关键参数**:
- `--device`: 录音设备ID（通过 `arecord -l` 查看）
- `--volume-threshold`: 触发录音的音量变化阈值(dB)，越小越灵敏
- `--silence-duration`: 判断说话结束的静音时长(秒)

## 常见问题与解决

### 1. Ollama无法连接
**现象**: 客户端报连接错误  
**解决**: 确保Ollama启动时绑定`0.0.0.0`，不是默认的`127.0.0.1`

```bash
OLLAMA_HOST=0.0.0.0 ollama serve
```

### 2. 无音频输出
**现象**: 程序运行但扬声器无声  
**解决**: Linux单板机Pygame驱动问题，脚本开头强制指定alsa

```python
import os
os.environ['SDL_AUDIODRIVER'] = 'alsa'
import pygame
```

### 3. VAD反应迟钝
**现象**: 说完话等很久才有反应  
**快速解决**: 降低 `--silence-duration` 到 0.6 或更小  
**精细调整**: 修改 `ASRClient.start_listening` 中的 `dynamic_silence_threshold` 和 `effective_threshold`

### 4. ASR模型加载失败
**现象**: SenseVoice服务启动报错  
**解决**: 
- 检查网络，首次会下载模型
- 确保有足够磁盘空间
- GPU内存不足时会自动降级到CPU

### 5. TTS声音质量差
**现象**: 合成语音不自然  
**解决**: 
- 准备高质量的参考音频文件（16kHz，wav格式）
- 调整index-tts-vllm的inference参数
- 确保参考音频时长在10-30秒

### 6. 命令行参数无效
**现象**: 修改参数后行为未改变  
**根本解决**: 直接修改对应类的`__init__`方法中的默认值

## 定制与扩展

### 更换LLM模型
```python
# LLMClient.__init__ 中修改
self.model = "qwen2.5:7b-instruct-q8_0"  # 或其他ollama模型
```

### 更换TTS声音
```python
# VoiceAssistant._process_conversation 中修改
self.tts_processor = TTSProcessor(
    audio_paths=["wavs/custom_voice.wav"],  # 你的参考音频
    # ...
)
```

### 适配其他服务
- **ASR**: 重写 `ASRClient.send_to_asr` 方法
- **TTS**: 重写 `TTSProcessor.generate_tts_audio` 方法
- **LLM**: 重写 `LLMClient` 的流式交互逻辑

### 性能优化建议
1. **ASR**: 调整batch_size和VAD参数
2. **TTS**: 启用vllm的tensor并行
3. **网络**: 使用千兆网络减少传输延迟
4. **硬件**: 服务端使用高性能GPU，客户端确保音频设备低延迟

## 系统架构

```
[麦克风] -> [客户端] --(音频)--> [ASR:8001] -> [LLM:11434] -> [TTS:11996] -> [客户端] -> [扬声器]
           香橙派                    SenseVoice    Qwen2.5      index-tts    音频流
```

---

1.联系方式: cialtion@outlook.com | cialtion737410@sjtu.edu.cn

2.Notes: https://haxxorcialtion.github.io/sub_htmls/局域网内香橙派完全本地ASR-LLM-TTS_Realtime架构.html
