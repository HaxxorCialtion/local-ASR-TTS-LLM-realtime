#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时语音助手 - 集成ASR-LLM-TTS的完整对话系统
(最终修复版：修正pygame初始化诊断代码)
"""

import os
import threading
import time
import queue
import subprocess
import requests
import tempfile
import wave
import signal
import sys
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import re
import math

# --- 核心修复：在导入pygame前，强制指定ALSA音频驱动 ---
os.environ['SDL_AUDIODRIVER'] = 'alsa'

try:
    import numpy as np
    import ollama
    import pygame
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("请安装: pip3 install numpy ollama pygame-ce")
    sys.exit(1)

# --- 增强初始化诊断 (已修正) ---
print("DEBUG: Initializing Pygame Mixer...")
pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
pygame.mixer.init()

if pygame.mixer.get_init():
    print(f"DEBUG: Pygame Mixer initialized successfully.")
    # 修正: get_driver() 只返回1个值
    driver_name = pygame.mixer.get_driver()
    # 修正: get_init() 返回3个值 (频率, 格式,声道数)
    freq, audio_format, channels = pygame.mixer.get_init()
    print(f"DEBUG: Driver: {driver_name}, Freq: {freq}, Channels: {channels}, Format: {audio_format}")
else:
    print("\nFATAL ERROR: Pygame Mixer FAILED to initialize. No audio will be played.")
    print("Please check your Orange Pi's audio configuration (e.g., using 'aplay -l').\n")
# --- ---

class KeyboardInterruptHandler:
    """在后台线程中监听键盘输入，以实现非阻塞中断"""
    def __init__(self):
        self.interrupt_event = threading.Event()
        self.thread = None
    def _listen_for_input(self):
        while True:
            try:
                input()
                self.interrupt_event.set()
            except EOFError:
                break
    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._listen_for_input, daemon=True)
            self.thread.start()
    def is_interrupted(self) -> bool:
        return self.interrupt_event.is_set()
    def reset(self):
        self.interrupt_event.clear()

@dataclass
class ConversationMetrics:
    """对话性能指标 (已升级为高精度分析模型)"""
    # 基础时间戳
    conversation_start: Optional[float] = None      # 开始录音（用户开始说话）
    user_speech_end_time: Optional[float] = None    # <--- 新增：用户最后声音的时间戳
    recording_stop_time: Optional[float] = None     # <--- 新增：客户端停止录音的时间戳
    asr_complete_time: Optional[float] = None       # ASR结果返回
    llm_first_token_time: Optional[float] = None    # LLM首个token
    llm_complete_time: Optional[float] = None       # LLM最后一个token
    tts_first_audio_time: Optional[float] = None    # TTS首段音频生成
    first_audio_play_time: Optional[float] = None   # 开始播放首段音频
    
    # 统计数据
    asr_text: str = ""
    llm_response: str = ""
    llm_total_tokens: int = 0
    tts_segments: int = 0
    audio_duration: float = 0.0

    def print_metrics(self):
        """打印性能指标 (已实现新的统计逻辑)"""
        if not self.conversation_start:
            return
            
        print("\n" + "="*50)
        print("📊 本轮对话性能指标")
        print("="*50)

        # --- 阶段一：用户输入阶段 ---
        if self.user_speech_end_time and self.conversation_start:
            speech_duration = self.user_speech_end_time - self.conversation_start
            print(f"🎤 用户说话时长: {speech_duration:.3f}s")
            print(f"📝 识别内容: {self.asr_text}")

        if self.recording_stop_time and self.user_speech_end_time:
            silence_latency = self.recording_stop_time - self.user_speech_end_time
            print(f"   - 🤫 静音监测延迟: {silence_latency:.3f}s (说完话后等待确认的时间)")

        if self.asr_complete_time and self.recording_stop_time:
            asr_processing_latency = self.asr_complete_time - self.recording_stop_time
            print(f"   - 🔎 ASR处理耗时: {asr_processing_latency:.3f}s (音频上传及云端识别)")

        print("-" * 20 + " 核心延迟 " + "-" * 22)
        
        # --- 阶段二：系统响应阶段 ---
        if self.first_audio_play_time and self.user_speech_end_time:
            core_latency = self.first_audio_play_time - self.user_speech_end_time
            print(f"⏱️ 核心交互延迟 (TTFA): {core_latency:.3f}s (从用户说完 -> 听到声音)")
        
        print("-" * 20 + " 延迟分解 " + "-" * 20)

        if self.llm_first_token_time and self.asr_complete_time:
            llm_first_token_latency = self.llm_first_token_time - self.asr_complete_time
            print(f"   - 🤖 client端 LLM首token延迟: {llm_first_token_latency:.3f}s")
        
        if self.llm_complete_time and self.llm_first_token_time:
            llm_generation_duration = self.llm_complete_time - self.llm_first_token_time
            print(f"   - ✍️  LLM token生成耗时: {llm_generation_duration:.3f}s")
            print(f"   - 🔤  生成tokens: {self.llm_total_tokens}")

        if self.tts_first_audio_time and self.llm_complete_time:
            tts_latency = self.tts_first_audio_time - self.llm_complete_time
            print(f"   - 🎵 client端 TTS首段音频延迟: {tts_latency:.3f}s")

        if self.first_audio_play_time and self.tts_first_audio_time:
            playback_queue_latency = self.first_audio_play_time - self.tts_first_audio_time
            print(f"   - 🔊 音频播放准备延迟: {playback_queue_latency:.3f}s")

        print("-" * 50)
        
        print(f"🎶 音频总时长: {self.audio_duration:.2f}s")
        total_time = time.time() - self.conversation_start
        print(f"⏱️ 端到端总耗时: {total_time:.3f}s")
        print("="*50)

class ASRClient:
    """ASR客户端模块 (最终完整版，包含高精度VAD和时间戳统计)"""
    def __init__(self, asr_url: str, device_id: int = None, 
                    volume_threshold: float = 6.0, silence_duration: float = 1.0):
            self.asr_url = asr_url
            self.device_id = device_id
            self.volume_threshold = volume_threshold
            self.silence_duration = silence_duration
            self.min_recording_duration = 1.0
            self.max_recording_duration = 30.0
            self.chunk_duration = 1.0  # 定义区块时长为1秒
            self.is_running = False
            self.is_paused = False
            self.state = "waiting"
            self.baseline_db = -40
            
            # --- 核心修复：使用math.ceil来正确计算缓冲区大小 ---
            buffer_size = math.ceil(self.silence_duration / self.chunk_duration)
            self.silence_buffer = deque(maxlen=int(buffer_size))
            
            self.recording_start_time = 0
            self.last_sound_time = 0
            self.peak_db_during_recording = -60.0
            self.on_text_callback: Optional[Callable[[str, float, float, float], None]] = None

    def set_text_callback(self, callback: Callable[[str, float, float, float], None]):
        """设置文本识别完成的回调函数"""
        self.on_text_callback = callback

    def start_listening(self):
        """开始语音监听"""
        self.is_running = True
        self.state = "waiting"
        print(f"🎤 ASR监听已启动")
        print(f"💡 音量增加 +{self.volume_threshold}dB 开始录音")
        print(f"💡 静音 {self.silence_duration}s 自动停止")
        
        recording_chunks = []
        recent_volumes = deque(maxlen=8)
        
        while self.is_running:
            if self.is_paused:
                time.sleep(0.1)
                continue
            
            try:
                temp_path = f"/tmp/detect_{int(time.time())}.wav"
                cmd = ['arecord','-D', f'plughw:{self.device_id}' if self.device_id else 'default','-d', '1','-f', 'S16_LE', '-r', '16000', '-c', '1',temp_path]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)

                if result.returncode != 0 or not os.path.exists(temp_path):
                    print(f"\r❌ 录音失败(code:{result.returncode})，重试中...", end="", flush=True)
                    if result.stderr:
                        print(f"\n[arecord error]: {result.stderr.strip()}")
                    time.sleep(1)
                    continue
                
                file_size = os.path.getsize(temp_path)
                if file_size < 1000:
                    os.unlink(temp_path)
                    continue
                
                with wave.open(temp_path, 'rb') as wav_file:
                    audio_data = wav_file.readframes(wav_file.getnframes())
                
                current_db = self.calculate_db(audio_data)
                recent_volumes.append(current_db)
                current_time = time.time()
                
                if len(recent_volumes) >= 4:
                    recent_avg = np.mean(list(recent_volumes)[-2:])
                    baseline_avg = np.mean(list(recent_volumes)[:-2])
                    volume_change = recent_avg - baseline_avg
                else:
                    volume_change = current_db - self.baseline_db
                
                if self.state == "waiting":
                    print(f"\r🔍 监听... {current_db:.1f}dB ({volume_change:+.1f})", end="", flush=True)
                    should_start = (volume_change > self.volume_threshold or (current_db > -25 and volume_change > 3) or current_db > -15)
                    if should_start:
                        self.state = "recording"
                        recording_chunks = [audio_data]
                        self.recording_start_time = current_time
                        self.last_sound_time = current_time
                        self.peak_db_during_recording = current_db # 重置峰值音量
                        self.silence_buffer.clear()
                        print(f"\n🔴 开始录音! {current_db:.1f}dB ({volume_change:+.1f})")

                elif self.state == "recording":
                    recording_chunks.append(audio_data)
                    recording_duration = current_time - self.recording_start_time
                    
                    if current_db > self.peak_db_during_recording:
                        self.peak_db_during_recording = current_db

                    dynamic_silence_threshold = self.peak_db_during_recording - 15  # 低于15dB开始静音
                    effective_threshold = max(-38.0, dynamic_silence_threshold)
                    is_sound_active = current_db > effective_threshold

                    if is_sound_active:
                        self.last_sound_time = current_time
                        self.silence_buffer.clear()
                        print(f"\r🔴 录音中... {recording_duration:.1f}s [{current_db:.1f}dB > {effective_threshold:.1f}dB]", end="", flush=True)
                    else:
                        self.silence_buffer.append(current_time)
                        silence_duration_so_far = current_time - self.last_sound_time
                        print(f"\r🔴 录音中... {recording_duration:.1f}s 静音{silence_duration_so_far:.1f}s [{current_db:.1f}dB <= {effective_threshold:.1f}dB]", end="", flush=True)
                    
                    should_stop = False
                    stop_reason = ""
                    if len(self.silence_buffer) >= self.silence_buffer.maxlen:
                        if recording_duration >= self.min_recording_duration:
                            should_stop = True; stop_reason = f"静音{self.silence_duration}s"
                    elif recording_duration >= self.max_recording_duration:
                        should_stop = True; stop_reason = "达到最大时长"
                    
                    if should_stop:
                        recording_stop_time = time.time()
                        
                        print(f"\n⏹️ 停止录音 ({stop_reason})")
                        self.state = "processing"
                        
                        threading.Thread(
                            target=self._process_recording,
                            args=(
                                recording_chunks.copy(), 
                                recording_duration, 
                                self.recording_start_time,
                                self.last_sound_time,
                                recording_stop_time
                            ),
                            daemon=True
                        ).start()
                        recording_chunks = []; self.silence_buffer.clear(); self.state = "waiting"

                elif self.state == "processing":
                    print(f"\r⏳ ASR识别中...", end="", flush=True)
                    time.sleep(0.1)
                
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except FileNotFoundError:
                print("\n\nFATAL ERROR: 'arecord' command not found."); print("Please make sure 'alsa-utils' is installed.")
                self.is_running = False; continue
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\r❌ ASR异常: {e}", end="", flush=True); time.sleep(0.5)
        
        print(f"\n🔇 ASR监听已停止")

    def _process_recording(self, audio_chunks: List[bytes], duration: float, start_time: float, last_sound_time: float, stop_time: float):
        """处理录音并识别"""
        try:
            audio_file = self.save_audio_chunks(audio_chunks)
            success, text = self.send_to_asr(audio_file)
            if success and text.strip():
                print(f"🎯 ASR识别结果: 『{text}』")
                if self.on_text_callback:
                    self.on_text_callback(text, start_time, last_sound_time, stop_time)
            else:
                print(f"❌ ASR识别失败: {text}")
            os.unlink(audio_file)
        except Exception as e:
            print(f"❌ ASR处理异常: {e}")
        finally:
            if self.state == "processing":
                self.state = "waiting"

    def pause_listening(self):
        """暂停监听"""
        if not self.is_paused:
            print("\n🔇 ASR监听已暂停 (音频播放中...)")
            self.is_paused = True

    def resume_listening(self):
        """恢复监听"""
        if self.is_paused:
            self.silence_buffer.clear()
            print("\n🎤 ASR监听已恢复")
            self.is_paused = False

    def calculate_db(self, audio_data: bytes) -> float:
        """计算音频分贝值"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            if len(audio_array) == 0: return -60
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            if rms < 1e-10: return -60
            db = 20 * np.log10(rms / 32767.0)
            return db
        except: return -60

    def send_to_asr(self, audio_file_path: str) -> tuple[bool, str]:
        """发送音频到ASR服务器"""
        try:
            with open(audio_file_path, 'rb') as f:
                files = {'file': ('audio.wav', f, 'audio/wav')}
                response = requests.post(f"{self.asr_url}/transcribe", files=files, timeout=60)
            if response.status_code == 200:
                result = response.json()
                if result.get('code') == 0:
                    data = result.get('data', {}); text = data.get('text', '').strip(); return True, text
                else: return False, result.get('msg', 'Unknown error')
            else: return False, f"HTTP {response.status_code}"
        except Exception as e: return False, str(e)

    def save_audio_chunks(self, audio_chunks: List[bytes]) -> str:
        """保存音频块为WAV文件"""
        temp_path = f"/tmp/recording_{int(time.time())}.wav"
        with wave.open(temp_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            for chunk in audio_chunks:
                if chunk: wav_file.writeframes(chunk)
        return temp_path

    def stop_listening(self):
        """停止监听"""
        self.is_running = False


class AudioPlayer:
    """音频播放器"""
    def __init__(self, metrics: ConversationMetrics):
        self.audio_queue = queue.Queue()
        self.playing = False
        self.player_thread = None
        self.metrics = metrics
        self.first_audio_played = False
    
    def _play_loop(self):
        while self.playing:
            try:
                audio_data, segment_id = self.audio_queue.get(timeout=0.1)
                duration = self._get_audio_duration(audio_data)
                temp_file = f"/tmp/play_{segment_id}_{int(time.time()*1000)}.wav"
                with open(temp_file, 'wb') as f: f.write(audio_data)
                
                if not self.first_audio_played:
                    if self.metrics.conversation_start:
                        self.metrics.first_audio_play_time = time.time()
                    self.first_audio_played = True
                try:
                    pygame.mixer.music.load(temp_file)
                    pygame.mixer.music.play()
                    time.sleep(0.05)
                    if not pygame.mixer.music.get_busy():
                        print(f"DEBUG: WARNING - Playback not busy immediately after play() call for segment {segment_id}.")
                    
                    print(f"🔊 播放段{segment_id} ({duration:.2f}s)")
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.01)
                    
                    self.metrics.audio_duration += duration
                    print(f"✅ 段{segment_id}播放完成")
                except Exception as e:
                    print(f"❌ 播放失败: {e}")
                try:
                    os.unlink(temp_file)
                except: pass
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ 播放器错误: {e}")

    def is_busy(self) -> bool:
        return pygame.mixer.music.get_busy() or not self.audio_queue.empty()
    def stop_current_playback(self):
        print("\n⏹️ 停止当前播放并清空队列...")
        pygame.mixer.music.stop()
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                continue
    def start_player(self):
        if not self.playing:
            self.playing = True
            self.player_thread = threading.Thread(target=self._play_loop, daemon=True)
            self.player_thread.start()
    def stop_player(self):
        self.playing = False
        if self.player_thread: self.player_thread.join()
    def add_audio(self, audio_data: bytes, segment_id: int = 0):
        self.audio_queue.put((audio_data, segment_id))
    def _get_audio_duration(self, audio_data: bytes) -> float:
        try:
            temp_file = f"/tmp/duration_{int(time.time()*1000)}.wav"
            with open(temp_file, 'wb') as f: f.write(audio_data)
            with wave.open(temp_file, 'rb') as wav_file:
                frames = wav_file.getnframes(); sample_rate = wav_file.getframerate()
                duration = frames / float(sample_rate)
            os.unlink(temp_file)
            return duration
        except: return 1.0

class TTSProcessor:
    """TTS处理器"""
    def __init__(self, tts_url: str, audio_paths: List[str], 
                 max_concurrent: int = 3, metrics: ConversationMetrics = None):
        self.tts_url = tts_url
        self.audio_paths = audio_paths
        self.max_concurrent = max_concurrent
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.metrics = metrics
        self.audio_player = AudioPlayer(metrics)
        self.audio_player.start_player()
        self.segment_counter = 0
        self.first_audio_generated = False
    def split_text_intelligently(self, text: str) -> List[str]:
        text = text.strip()
        if not text: return []
        sentences = re.split(r'([，。！？；、,!?;\n])', text)
        segments = []
        if len(sentences) > 1:
            for i in range(0, len(sentences) - 1, 2):
                segment = (sentences[i] + sentences[i+1]).strip()
                if segment: segments.append(segment)
            last_part = sentences[-1].strip()
            if last_part: segments.append(last_part)
        else:
            segments.append(text)
        print(f"📊 分段结果 ({len(segments)}段): {segments}")
        return segments
    def generate_tts_audio(self, text: str, segment_id: int) -> Optional[bytes]:
        try:
            payload = { "text": text, "audio_paths": self.audio_paths }
            response = requests.post(self.tts_url, json=payload, timeout=15)
            if response.status_code == 200:
                print(f"✅ 段{segment_id} TTS完成 ({len(response.content)} bytes)")
                return response.content
            else:
                print(f"❌ 段{segment_id} TTS失败: HTTP {response.status_code}"); return None
        except Exception as e:
            print(f"❌ 段{segment_id} TTS异常: {e}"); return None
    def process_text(self, text: str):
        segments = self.split_text_intelligently(text)
        if not segments:
            print("⚠️ TTS警告: 没有可处理的文本段。"); return
        self.metrics.tts_segments = len(segments)
        first_segment = segments.pop(0)
        self.segment_counter += 1
        first_id = self.segment_counter
        print(f"🚀 优先处理第一段: '{first_segment}'")
        try:
            audio_data = self.generate_tts_audio(first_segment, first_id)
            if audio_data:
                if not self.first_audio_generated and self.metrics.conversation_start:
                    self.metrics.tts_first_audio_time = time.time()
                    self.first_audio_generated = True
                self.audio_player.add_audio(audio_data, first_id)
                print(f"🎯 第一段已提交播放")
        except Exception as e:
            print(f"❌ 第一段处理异常: {e}")
        if segments:
            futures = []
            for segment in segments:
                self.segment_counter += 1
                segment_id = self.segment_counter
                future = self.executor.submit(self.generate_tts_audio, segment, segment_id)
                futures.append((future, segment_id))
            for future, segment_id in futures:
                try:
                    audio_data = future.result(timeout=20)
                    if audio_data:
                        self.audio_player.add_audio(audio_data, segment_id)
                        print(f"✅ 段{segment_id}已提交播放")
                except Exception as e:
                    print(f"❌ 段{segment_id}处理失败: {e}")
    def cleanup(self):
        self.executor.shutdown(wait=True); self.audio_player.stop_player()

class LLMClient:
    """LLM客户端"""
    def __init__(self, model: str = "qwen2.5:7b-instruct-q8_0", ollama_host: str = "localhost:11434"):
        self.model = model; self.ollama_host = ollama_host
        self.messages = [{"role": "system", "content": "你的名字是纳西妲，你在和旅者口语化聊天。"}]
        self.client = ollama.Client(host=f'http://{ollama_host}')
    def chat(self, user_input: str, metrics: ConversationMetrics) -> str:
        try:
            self.messages.append({"role": "user", "content": user_input})
            print("思考中...")
            full_response = ""; first_token_received = False; token_count = 0
            stream = self.client.chat(model=self.model, messages=self.messages, stream=True)
            for chunk in stream:
                current_time = time.time()
                if not first_token_received:
                    if metrics.conversation_start:
                        metrics.llm_first_token_time = current_time
                    first_token_received = True
                    print(f"LLM首token延迟: {metrics.llm_first_token_time - metrics.conversation_start:.3f}s")
                delta = chunk.get('message', {}).get('content', '')
                if delta:
                    full_response += delta; token_count += 1
                    print(delta, end='', flush=True)
            print()
            if metrics.conversation_start:
                metrics.llm_complete_time = time.time()
            metrics.llm_total_tokens = token_count; metrics.llm_response = full_response
            print(f"LLM响应完成 ({token_count} tokens)")
            self.messages.append({"role": "assistant", "content": full_response})
            if len(self.messages) > 20:
                self.messages = self.messages[:1] + self.messages[-19:]
            return full_response
        except Exception as e:
            print(f"LLM处理失败: {e}"); return ""

class VoiceAssistant:
    """语音助手主控制器 (已更新回调逻辑以匹配ASRClient)"""
    def __init__(self, asr_url: str, tts_url: str, asr_device: int = None, ollama_host: str = "localhost:11434"):
        self.asr_url, self.tts_url, self.asr_device, self.ollama_host = asr_url, tts_url, asr_device, ollama_host
        self.asr_client = ASRClient(asr_url, asr_device)
        self.llm_client = LLMClient(ollama_host=ollama_host)
        self.tts_processor = None
        self.interrupt_handler = KeyboardInterruptHandler()
        self.is_running = False; self.current_metrics = None; self.conversation_active = False
        self.asr_client.set_text_callback(self._on_asr_result)

    def _on_asr_result(self, text: str, asr_start_time: float, last_sound_time: float, recording_stop_time: float):
        """ASR识别完成回调 (已更新，接收所有时间戳)"""
        if not text.strip(): return
        print(f"\n👤 用户: {text}")
        
        self.current_metrics = ConversationMetrics()
        
        # 将所有精确时间戳存入指标对象
        self.current_metrics.conversation_start = asr_start_time
        self.current_metrics.user_speech_end_time = last_sound_time
        self.current_metrics.recording_stop_time = recording_stop_time
        self.current_metrics.asr_complete_time = time.time() # ASR完成时间以收到回调为准
        self.current_metrics.asr_text = text
        
        self.conversation_active = True
        threading.Thread(target=self._process_conversation, args=(text,), daemon=True).start()

    def _process_conversation(self, user_text: str):
        self.asr_client.pause_listening()
        try:
            self.tts_processor = TTSProcessor(tts_url=self.tts_url, audio_paths=["wavs/纳西妲.wav"], max_concurrent=3, metrics=self.current_metrics)
            print("🤖 助手: ", end="", flush=True)
            llm_response = self.llm_client.chat(user_text, self.current_metrics)
            if llm_response:
                print(f"\n🎵 开始语音合成...")
                self.tts_processor.process_text(llm_response)
                self.interrupt_handler.reset()
                print("🎧 等待音频播放完成... (在终端按 Enter 键可跳过)")
                while self.tts_processor.audio_player.is_busy():
                    if self.interrupt_handler.is_interrupted():
                        print("\n⏩ 用户跳过播放")
                        self.tts_processor.audio_player.stop_current_playback(); break
                    time.sleep(0.1)
                if not self.interrupt_handler.is_interrupted():
                    print("✅ 所有音频播放完毕")
                self.current_metrics.print_metrics()
        except Exception as e:
            print(f"❌ 对话处理异常: {e}"); import traceback; traceback.print_exc()
        finally:
            if self.tts_processor:
                self.tts_processor.cleanup()
            self.conversation_active = False
            time.sleep(0.2)
            self.asr_client.resume_listening()

    def test_services(self) -> bool:
        print("🔧 测试服务连接...")
        try:
            response = requests.get(f"{self.asr_url}/health", timeout=5)
            if response.status_code == 200:
                result = response.json()
                if result.get('model_loaded'): print("✅ ASR服务正常")
                else: print("⚠️ ASR模型未加载"); return False
            else: print("❌ ASR服务异常"); return False
        except Exception as e: print(f"❌ ASR连接失败: {e}"); return False
        try:
            test_payload = {"text": "测试", "audio_paths": ["wavs/纳西妲.wav"]}
            response = requests.post(self.tts_url, json=test_payload, timeout=10)
            if response.status_code == 200: print("✅ TTS服务正常")
            else: print("❌ TTS服务异常"); return False
        except Exception as e: print(f"❌ TTS连接失败: {e}"); return False
        try:
            test_client = ollama.Client(host=f'http://{self.ollama_host}')
            test_client.chat(model=self.llm_client.model, messages=[{"role": "user", "content": "测试"}])
            print("✅ LLM服务正常")
        except Exception as e:
            print(f"❌ LLM连接失败: {e}"); print(f"   请确保ollama在 {self.ollama_host} 上运行"); return False
        return True

    def start(self):
        print("🌟 实时语音助手启动"); print("="*40)
        if not self.test_services():
            print("❌ 服务测试失败")
            choice = input("是否继续运行? (y/N): ").lower()
            if choice != 'y': return
        self.interrupt_handler.start()
        print("\n🎤 开始语音监听..."); print("💡 对着设备说话即可开始对话"); print("💡 按 Ctrl+C 退出程序")
        self.is_running = True
        try:
            self.asr_client.start_listening()
        except KeyboardInterrupt:
            print("\n\n👋 程序退出")
        finally:
            self.stop()

    def stop(self):
        print("🛑 正在停止服务...")
        self.is_running = False
        self.asr_client.stop_listening()
        if self.tts_processor: self.tts_processor.cleanup()
        print("✅ 服务已停止")
def main():
    import argparse
    parser = argparse.ArgumentParser(description='实时语音助手')
    parser.add_argument('--asr-url', required=True, help='ASR服务器地址')
    parser.add_argument('--tts-url', default='http://127.0.0.1:11996/tts_url', help='TTS服务器地址')
    parser.add_argument('--ollama-host', default='localhost:11434', help='Ollama服务器地址')
    parser.add_argument('--device', type=int, help='录音设备ID')
    parser.add_argument('--volume-threshold', type=float, default=6.0, help='音量检测阈值')
    parser.add_argument('--silence-duration', type=float, default=0.8, help='静音停止时长')
    args = parser.parse_args()
    assistant = VoiceAssistant(
        asr_url=args.asr_url, tts_url=args.tts_url,
        asr_device=args.device, ollama_host=args.ollama_host
    )
    assistant.asr_client.volume_threshold = args.volume_threshold
    assistant.asr_client.silence_duration = args.silence_duration
    assistant.start()

if __name__ == "__main__":
    main()