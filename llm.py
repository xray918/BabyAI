from openai import OpenAI
from dotenv import load_dotenv
import os
import time
from log import logger
import io
import random

MODEL_LIST = ["gpt-3.5-turbo", "gpt-4o", "gpt-4", "wenxin", "wenxin-4", "xunfei", "claude", "gpt-4-turbo",
              "gpt-4-turbo-preview", "gpt-4-1106-preview", "moonshot-v1-32k", "qwen-turbo", "gemini"]

# 加载 .env 文件
load_dotenv()

def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class LLM(object):
    def __init__(self):
        self.models = {}
        
        # OpenAI API配置
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = "http://47.251.57.51:5000"
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
        
        # 注册模型
        self.models["gpt-3.5-turbo"] = True
        self.models["gpt-4o"] = True
        self.models["gpt-4o-mini"] = True
        self.models["whisper-1"] = True
        self.models["tts-1"] = True

    def text_to_voice(self, text):
        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=text
            )
            file_name = f"tmp/{random.randint(0, 100000)}.mp3"
            logger.info(f"[OPENAI] text_to_Voice file_name={file_name}, input={text}")
            response.stream_to_file(file_name)
            logger.info(f"[OPENAI] text_to_Voice success")
            return {"type": "voice", "content": file_name}
        except Exception as e:
            logger.error(e)
            return {"type": "error", "content": "遇到了一点小问题，请稍后再问我吧"}

    def voice_to_text(self, voice_file, **kwargs):
        model = kwargs.get("model", "whisper-1")
        
        if model in self.models:
            try:
                with open(voice_file, "rb") as file:
                    text = self.client.audio.transcriptions.create(
                        model=model,
                        file=file,
                        response_format="text"
                    )
                logger.info(f"[LLM] voiceToText text={text} voice file name={voice_file}")
                return {"type": "text", "content": text}
            except Exception as e:
                logger.warning(f"openai voice transcribe err:{e}")
                return {"type": "error", "content": "我暂时还无法听清您的语音，请稍后再试吧~"}
        else:
            logger.warning(f"model:{model} is not in supported list")
            return None

    def voice_raw_to_text(self, audio, **kwargs):
        model = kwargs.get("model", "whisper-1")
        
        if model in self.models:
            try:
                with open("test.wav", "wb") as f:
                    f.write(audio.get_wav_data())
                
                with open("test.wav", "rb") as file:
                    text = self.client.audio.transcriptions.create(
                        model=model,
                        file=file,
                        response_format="text",
                        temperature=0,
                        language="zh"
                    )
                
                if not text or text == '\n':
                    raise ValueError("没有识别出人声")
                
                return text
            except ValueError as e:
                raise e
            except Exception as e:
                logger.warning(f"openai voice transcribe err: {e}")
                return ""
        else:
            logger.warning(f"model: {model} is not in supported list")
            return None

    def reply_text(self, messages, **kwargs):
        model = kwargs.get("model", "gpt-4o-mini")
        
        try:
            if model not in self.models:
                model = "gpt-4o-mini"
                logger.warning(f"model:{model} not in model_list")

            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.3),
                stream=kwargs.get("stream", False)
            )
            
            if not kwargs.get("stream", False):
                return {
                    "total_tokens": response.usage.total_tokens,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "content": response.choices[0].message.content
                }
            return response
        except Exception as e:
            logger.exception(f"[ChatGPT] Exception: {e}")
            return {"completion_tokens": 0, "content": str(e)}

    def reply_text_stream(self, messages, **kwargs):
        kwargs["stream"] = True
        return self.reply_text(messages, **kwargs)
