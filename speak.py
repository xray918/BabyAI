import pyaudio
from pyaudio import PyAudio, paInt16
import speech_recognition as sr
import queue
import threading
import httpx
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
        self.stream = self.p.open(frames_per_buffer=1000, format=pyaudio.paInt16, channels=1, rate=44100, output=True)

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
            # 播放音频块
            self.stream.write(audio_chunk)
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

            request = ServeTTSRequest(
                text=text,
                reference_id="8d8db016275a4ad08be54277aa982954",
                latency="normal",
                format="pcm",
                chunk_length=200,
            )
            print("[TTS] Sending request to fish.audio API...")
            with httpx.Client() as client:
                response = client.post(
                    "https://api.fish.audio/v1/tts",
                    content=ormsgpack.packb(request, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
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


    def _play_worker(self):
        while True:
            try:
                msg = self.audio_queue.get(block=False)
            except queue.Empty:
                # 如果队列为空，播放 MP3 文件
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
                audio_chunk = msg.get("payload")
                print(f"[PLAY] Got audio chunk, size: {len(audio_chunk) if audio_chunk else 0} bytes")

                if audio_chunk is None:
                    print("[PLAY] Audio chunk is None, breaking")
                    break

                try:
                    self.stream.write(audio_chunk)
                    print("[PLAY] Successfully played audio chunk")
                except Exception as e:
                    print(f"[PLAY] Error playing audio chunk: {e}")

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
                self.stream.write(wav_buf)

    def close(self):

        self.text_queue.put({"type": "flag", "flag": "exit"})
        # self.audio_queue.put({"type": "flag", "flag": "over"})
        self.speak_thread.join()
        self.play_thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
