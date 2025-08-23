#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版 SenseVoice ASR 服务端 - 专门用于WAV文件速度测试
去掉复杂的优化逻辑，专注最低延迟
"""

import asyncio
import json
import logging
import argparse
import numpy as np
import torch
import uvicorn
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import wave
import io
import tempfile
import os

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleSenseVoiceASR:
    """简化版SenseVoice ASR引擎 - 专注速度"""

    def __init__(self, model_path: str = "./asr_models"):
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")

    def load_model(self):
        """加载模型"""
        try:
            logger.info(f"正在加载 SenseVoice 模型从: {self.model_path}")
            start_time = time.time()

            # 导入FunASR
            from funasr import AutoModel

            # 设备字符串
            device_str = "cuda:0" if self.device.type == "cuda" else "cpu"

            self.model = AutoModel(
                model=self.model_path,
                trust_remote_code=True,
                # 根据官方示例添加VAD
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                device=device_str,
                disable_update=True
            )

            load_time = time.time() - start_time
            logger.info(f"模型加载完成，耗时: {load_time:.2f}秒")
            return True

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False

    def transcribe_wav(self, wav_file_path: str) -> dict:
        """转录WAV文件"""
        try:
            start_time = time.time()

            # 检查文件是否存在
            if not os.path.exists(wav_file_path):
                raise FileNotFoundError(f"文件不存在: {wav_file_path}")

            logger.info(f"开始处理文件: {wav_file_path}")

            # 根据官方示例调用模型
            result = self.model.generate(
                input=wav_file_path,
                cache={},
                language="auto",  # 自动检测语言，也可以指定 "zh", "en" 等
                use_itn=True,  # 逆文本标准化
                batch_size_s=60,  # 批处理大小（秒）
                merge_vad=True,  # 合并VAD结果
            )

            process_time = time.time() - start_time

            # 处理结果
            if result and len(result) > 0:
                # 使用官方后处理函数
                try:
                    from funasr.utils.postprocess_utils import rich_transcription_postprocess
                    text = rich_transcription_postprocess(result[0]["text"])
                except ImportError:
                    # 如果没有后处理函数，直接使用原始文本
                    text = result[0].get("text", "").strip()

                # 获取音频时长信息
                audio_duration = self._get_audio_duration(wav_file_path)

                response = {
                    "success": True,
                    "text": text,
                    "audio_duration": audio_duration,
                    "process_time": process_time,
                    "rtf": process_time / audio_duration if audio_duration > 0 else 0,  # 实时因子
                    "timestamp": time.time()
                }

                logger.info(
                    f"处理完成 - 音频时长: {audio_duration:.2f}s, 处理时长: {process_time:.2f}s, RTF: {response['rtf']:.3f}")
                logger.info(f"识别结果: {text}")

                return response
            else:
                return {
                    "success": False,
                    "error": "模型返回空结果",
                    "process_time": process_time
                }

        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"转录失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "process_time": error_time
            }

    def _get_audio_duration(self, wav_file_path: str) -> float:
        """获取音频时长"""
        try:
            with wave.open(wav_file_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                duration = frames / float(sample_rate)
                return duration
        except:
            return 0.0


from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    logger.info("正在启动 SenseVoice ASR 服务...")
    success = asr_engine.load_model()
    if not success:
        logger.error("模型加载失败!")
        raise RuntimeError("模型加载失败")
    else:
        logger.info("SenseVoice ASR 服务启动成功!")

    yield

    # 关闭时执行
    logger.info("正在关闭服务...")


# FastAPI应用
app = FastAPI(
    title="简化版 SenseVoice ASR",
    version="1.0.0",
    lifespan=lifespan
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局ASR实例
asr_engine = SimpleSenseVoiceASR()


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """转录音频文件"""
    if not asr_engine.model:
        raise HTTPException(status_code=503, detail="模型未加载")

    # 检查文件类型
    if not file.filename.lower().endswith('.wav'):
        raise HTTPException(status_code=400, detail="只支持WAV文件")

    try:
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # 进行转录
        result = asr_engine.transcribe_wav(tmp_file_path)

        # 清理临时文件
        os.unlink(tmp_file_path)

        return {
            "code": 0 if result["success"] else -1,
            "msg": "success" if result["success"] else result.get("error", "unknown error"),
            "data": result
        }

    except Exception as e:
        logger.error(f"文件处理错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe_local")
async def transcribe_local_file(file_path: str):
    """转录本地文件 - 用于测试"""
    if not asr_engine.model:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        result = asr_engine.transcribe_wav(file_path)
        return {
            "code": 0 if result["success"] else -1,
            "msg": "success" if result["success"] else result.get("error", "unknown error"),
            "data": result
        }
    except Exception as e:
        logger.error(f"本地文件处理错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": asr_engine.model is not None,
        "device": str(asr_engine.device)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行简化版 SenseVoice ASR 服务")
    parser.add_argument('--port', type=int, default=8001, help='服务端口')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='服务主机')
    parser.add_argument('--model-path', type=str, default="./asr_models", help='模型路径')

    args = parser.parse_args()

    # 更新模型路径
    asr_engine.model_path = args.model_path

    logger.info(f"启动服务在 {args.host}:{args.port}")
    logger.info(f"模型路径: {args.model_path}")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )