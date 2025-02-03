import pyaudio
from pyaudio import PyAudio, paInt16
import speech_recognition as sr
import queue
import threading
import httpx
import numpy as np
import sounddevice as sd
import webrtcvad
from array import array
from collections import deque

import ormsgpack
from pydantic import BaseModel, conint
from typing import Annotated, Literal
from datetime import datetime
import time
# from playsound import playsound
from pydub import AudioSegment
# 初始化语音识别
recognizer = sr.Recognizer()
def print_time(pos):
    now_datetime = datetime.now()
    print(f"{pos},time:", now_datetime.strftime("%H:%M:%S.%f"))

# from devices import AudioDevices

# FORMAT = paInt16
# CHANNELS = 1
# SAMPLE_WIDTH = 2  # PyAudio.get_sample_size(pyaudio, format=paInt16)
# CHUNK_SIZE = 1024

class ServeReferenceAudio(BaseModel):
    audio: bytes
    text: str

class ServeTTSRequest(BaseModel):
    text: str
    chunk_length: Annotated[int, conint(ge=100, le=300, strict=True)] = 200
    format: Literal["wav", "pcm", "mp3"] = "mp3"
    mp3_bitrate: Literal[64, 128, 192] = 128
    references: list[ServeReferenceAudio] = []
    reference_id: str | None = None
    normalize: bool = True
    latency: Literal["normal", "balanced"] = "balanced"

def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance
@singleton
class StreamSpeak:

    # lock = threading.Lock()  # 用于线程安全
    def __init__(self):
        self.p = PyAudio()
        
        # 修改采样率为 VAD 支持的值
        self.RATE = 48000  # 改为 48kHz
        self.CHANNELS = 1
        self.CHUNK = 2048
        self.BUFFER_DURATION = 0.2
        self.BUFFER_SIZE = int(self.RATE * self.BUFFER_DURATION)
        
        # 创建音频缓冲区，添加淡入淡出
        self.audio_buffer = deque(maxlen=self.BUFFER_SIZE)
        self.fade_samples = int(0.01 * self.RATE)
        
        # 初始化 VAD，降低灵敏度
        self.vad = webrtcvad.Vad(2)
        
        # 分别创建输入和输出流
        self.input_stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self._input_callback
        )
        
        self.output_stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=self.CHANNELS,
            rate=self.RATE,
            output=True,
            frames_per_buffer=self.CHUNK
        )

        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()

        self.is_speaking = False
        self.is_thinking = False

        # 加载音频文件到内存
        self.audio_files = {
            'popo': self.load_audio('popolg.mp3'),
            'da': self.load_audio('da.mp3')
        }

        self.speak_thread = threading.Thread(target=self._speak_worker, daemon=True)
        self.play_thread = threading.Thread(target=self._play_worker, daemon=True)
        self.speak_thread.start()
        self.play_thread.start()


    def load_audio(self, file_path):
        # 使用 pydub 加载音频文件
        audio = AudioSegment.from_file(file_path)
        # 转换为适合 PyAudio 播放的格式
        return audio.raw_data

    def play_audio(self, audio_key):
        if audio_key in self.audio_files:
            audio_chunk = self.audio_files[audio_key]
            # 添加淡入淡出效果
            audio_data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            result = np.zeros_like(audio_data)
            result[:] = audio_data[:]
            
            if len(result) > self.fade_samples * 2:
                fade_in = np.linspace(0, 1, self.fade_samples)
                fade_out = np.linspace(1, 0, self.fade_samples)
                result[:self.fade_samples] = audio_data[:self.fade_samples] * fade_in
                result[-self.fade_samples:] = audio_data[-self.fade_samples:] * fade_out
            
            self.output_stream.write(result.tobytes())
        else:
            print(f"Audio file '{audio_key}' not found.")

    def get_speaking_status(self):
        return self.is_speaking

    def set_speaking_active(self,active):
        self.is_speaking = active

    def _speak_worker(self):
        while True:
            msg = self.text_queue.get()
            if msg is None:
                print("speak worker异常")
                time.sleep(1)
                continue
            elif msg.get("type") == "flag":
                if msg.get("flag") == "exit":
                    self._synthesize(msg)
                    self.text_queue.task_done()
                    return

            self._synthesize(msg)

            self.text_queue.task_done()

    def _synthesize(self, msg):
        try:
            if msg.get("type") == "payload":
                text = msg.get("payload","")
                print(f"[TTS] Converting text to speech: {text}")
            else:
                self.audio_queue.put(msg)
                return

            # 将 ServeTTSRequest 转换为字典
            request = ServeTTSRequest(
                text=text,
                reference_id="8d8db016275a4ad08be54277aa982954",
                latency="normal",
                format="pcm",
                chunk_length=200,
            )
            
            # 创建请求字典
            request_dict = {
                "text": request.text,
                "reference_id": request.reference_id,
                "latency": request.latency,
                "format": request.format,
                "chunk_length": request.chunk_length,
                "references": [],
                "normalize": request.normalize,
                "mp3_bitrate": request.mp3_bitrate
            }

            print("[TTS] Sending request to fish.audio API...")
            with httpx.Client() as client:
                response = client.post(
                    "https://api.fish.audio/v1/tts",
                    content=ormsgpack.packb(request_dict),  # 使用字典而不是 Pydantic 模型
                    headers={
                        "Authorization": "Bearer 7d768b26025644edbf022ce9bfb3d425",
                        "Content-Type": "application/msgpack",
                    },
                    timeout=None,
                )
                
                print(f"[TTS] API Response status: {response.status_code}")
                if response.status_code != 200:
                    print(f"[TTS] API Error: {response.text}")
                    return
                
                audio_data = response.content
                print(f"[TTS] Received audio data, size: {len(audio_data)} bytes")
                self.audio_queue.put({"type": "payload", "payload": audio_data})
                self.is_speaking = True
                
        except Exception as e:
            print(f"[TTS] Error in speak worker: {str(e)}")
            print(f"[TTS] Error type: {type(e)}")
            import traceback
            print(f"[TTS] Stack trace: {traceback.format_exc()}")

    def _input_callback(self, in_data, frame_count, time_info, status):
        """只处理输入音频的回调函数"""
        try:
            # 将输入音频转换为numpy数组
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            processed_data = np.zeros_like(audio_data)
            processed_data[:] = audio_data[:]
            
            # 回声消除处理
            if len(self.audio_buffer) > 0:
                echo_sample = np.array(list(self.audio_buffer)[-len(audio_data):])
                if len(echo_sample) == len(audio_data):
                    fade_in = np.linspace(0, 1, self.fade_samples)
                    fade_out = np.linspace(1, 0, self.fade_samples)
                    
                    result = np.zeros_like(processed_data)
                    result[:] = processed_data[:]
                    
                    if len(result) > self.fade_samples * 2:
                        # 应用淡入淡出
                        result[:self.fade_samples] = processed_data[:self.fade_samples] * fade_in
                        result[-self.fade_samples:] = processed_data[-self.fade_samples:] * fade_out
                        
                        # 减小回声消除的强度
                        echo_reduction = 0.15
                        result = result - (echo_reduction * echo_sample)
                        
                        # 应用噪声门限，使用更小的阈值
                        noise_gate = 0.005
                        noise_mask = np.abs(result) >= noise_gate
                        result = result * noise_mask
                        
                        # 应用平滑处理
                        window_size = 5
                        result = np.convolve(result, np.ones(window_size)/window_size, mode='same')
                        
                        # 检测语音并调整音量
                        is_speech = self._is_speech(result)
                        if is_speech and self.is_speaking:
                            reduction = 0.6
                            volume_reduced = np.zeros_like(result)
                            volume_reduced[:] = result[:] * reduction
                            volume_reduced[:self.fade_samples] *= fade_in
                            volume_reduced[-self.fade_samples:] *= fade_out
                            result = volume_reduced
                        
                        return (result.tobytes(), pyaudio.paContinue)
            
            return (processed_data.tobytes(), pyaudio.paContinue)
            
        except Exception as e:
            print(f"Error in input callback: {e}")
            return (in_data, pyaudio.paContinue)

    def _is_speech(self, audio_data):
        """使用 WebRTC VAD 检测是否有语音"""
        try:
            # 将float32转换为int16，确保数据范围正确
            audio_data_int = np.clip(audio_data * 32768, -32768, 32767).astype(np.int16)
            
            # VAD需要16位PCM数据
            frame_duration = 20  # ms，使用较短的帧长度
            samples_per_frame = int(self.RATE * frame_duration / 1000)
            
            # 确保数据长度满足要求
            if len(audio_data_int) >= samples_per_frame:
                # 只取需要的样本数
                frame_data = audio_data_int[:samples_per_frame]
                frame = frame_data.tobytes()
                try:
                    return self.vad.is_speech(frame, self.RATE)
                except Exception as e:
                    # 如果单个帧处理失败，返回 False 而不是抛出异常
                    return False
            
            return False
        except Exception as e:
            print(f"Error in speech detection: {e}")
            return False

    def _play_worker(self):
        while True:
            try:
                msg = self.audio_queue.get(block=False)
            except queue.Empty:
                if self.is_thinking:
                    self.play_audio('popo')
                else:
                    time.sleep(0.3)
                continue
            except Exception as e:
                print(f"[PLAY] Error in play worker: {e}")
                continue

            msg_type = msg.get("type")
            print(f"[PLAY] Processing message type: {msg_type}")

            if msg_type == "payload":
                audio_data = msg.get("payload")
                if audio_data:
                    # 转换音频数据为float32格式
                    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # 存储到缓冲区用于回声消除
                    self.audio_buffer.extend(audio_array)
                    
                    # 使用输出流播放
                    self.output_stream.write(audio_array.tobytes())

            elif msg_type == "flag":
                flag = msg.get("flag")
                print(f"[PLAY] Processing flag: {flag}")
                if flag == "over":
                    self.is_speaking = False
                    self.is_thinking = False
                    self.play_audio('da')
                elif flag == "listen_over":
                    self.is_thinking = True
                elif flag == "speak_start":
                    self.is_thinking = False
                elif flag == "exit":
                    self.audio_queue.task_done()
                    print("[PLAY] Received exit flag")
                    return
            else:
                print(f"[PLAY] Undefined message type: {msg_type}")

            self.audio_queue.task_done()

    def speak_and_play(self,text):
        request = ServeTTSRequest(
            text=text,
            reference_id="7f40adb65e394374b7041ad27b43e0c2",
            latency="balanced",
            format="pcm",
            chunk_length=200,
        )
        with httpx.Client() as client:
            with client.stream(
                    "POST",
                    "https://api.fish.audio/v1/tts",
                    content=ormsgpack.packb(request, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
                    headers={
                        "authorization": "7d768b26025644edbf022ce9bfb3d425",
                        "content-type": "application/msgpack",
                    },
                    timeout=None,
            ) as response:
                wav_buf =b''
                for chunk in response.iter_bytes(10000):  # 增加每次读取的字节数
                    wav_buf = wav_buf + chunk
                print(f"播放...")
                self.output_stream.write(wav_buf)

    def close(self):
        self.text_queue.put({"type": "flag", "flag": "exit"})
        self.speak_thread.join()
        self.play_thread.join()
        
        # 关闭所有流
        self.input_stream.stop_stream()
        self.input_stream.close()
        self.output_stream.stop_stream()
        self.output_stream.close()
        
        self.p.terminate()
        self.audio_buffer.clear()
