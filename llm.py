from dotenv import load_dotenv
import os
import time
from openai import OpenAI
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
        self.models = {
        }
        # qwen

        # kimi
        api_key =  os.getenv("OPENAI_API_KEY")
        api_base = "http://47.251.57.51:5000"
        client = OpenAI(api_key=api_key,base_url=api_base)

        self.models["gpt-3.5-turbo"] = client
        self.models["gpt-4o"] = client
        self.models["gpt-4o-mini"] = client
        self.models["whisper-1"] = client
        self.models["tts-1"] = client

    def text_to_voice(self,text):
        client = self.models["tts-1"]
        try:
            response = client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=text
            )
            file_name = "tmp/" + str(random.randint(0, 100000)) + ".mp3"
            logger.info(f"[OPENAI] text_to_Voice file_name={file_name}, input={text}")
            response.stream_to_file(file_name)
            logger.info(f"[OPENAI] text_to_Voice success")
            reply = Reply(ReplyType.VOICE, file_name)

        except Exception as e:
            logger.error(e)
            reply = Reply(ReplyType.ERROR, "遇到了一点小问题，请稍后再问我吧")

        return reply
    def voice_to_text(self,voice_file,**kwargs):
        model = kwargs.get("model","whisper - 1")

        if model in self.models:
            client = self.models[model]
            logger.info("[Openai] voice file name={}".format(voice_file))
            try:
                file = open(voice_file, "rb")
                text = client.audio.transcriptions.create(model=model, file=file,
                                                                 response_format="text")

                reply = Reply(ReplyType.TEXT, text)
                logger.info("[LLM] voiceToText text={} voice file name={}".format(text, voice_file))
            except Exception as e:
                logger.warning(f"openai vocie transcribe err:{e}")
                reply = Reply(ReplyType.ERROR, "我暂时还无法听清您的语音，请稍后再试吧~")
            finally:
                return reply
        else:
            logger.warning(f"mode:{model} is not in supported list")
            return None

    def voice_raw_to_text(self, audio, **kwargs):
        model = kwargs.get("model", "whisper-1")
        reply = ""
        if model in self.models:
            client = self.models[model]
            # logger.info("[Openai] processing audio input")
            try:
                with open("test.wav", "wb") as f:
                    f.write(audio.get_wav_data())

                file = open("test.wav", "rb")
                # # Convert the audio data to a file-like object

                text = client.audio.transcriptions.create(model=model, file=file,response_format="text",temperature=0,language="zh")
                if not text or text == '\n':
                    raise ValueError("没有识别出人声")

                reply = text
                # logger.info("[LLM] voiceToText text={}".format(text))
            except ValueError as e:
                raise e
            except Exception as e:
                logger.warning(f"openai voice transcribe err: {e}")
                reply = ""

            return reply
            # finally:
            #     return reply
        else:
            logger.warning(f"model: {model} is not in supported list")
            return None

    def reply_text(self,messages,**kwargs) -> dict:
        model = kwargs.get("model","qwen-turbo")
        try:

            if model not in self.models:
                model = "gpt-4o-mini"
                logger.warning(f"model:{model} not in model_list")

            client = self.models[model]
            logger.debug(f"llm args:{kwargs}")

            response = client.chat.completions.create(
                model=model,
                messages=messages,  # 假设session.messages是已正确格式化的消息列表
                temperature=kwargs.get("temperature", 0.3),
                # max_tokens=kwargs.get("max_tokens", 2000),
                stream = kwargs.get("stream",False),
            )
            return {
                "total_tokens": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "content": response.choices[0].message.content,
            }

        except Exception as e:
                logger.exception("[ChatGPT] Exception: {}".format(e))
                return {"completion_tokens": 0, "content": e}


    def reply_text_stream(self, messages, **kwargs) -> dict:
        model = kwargs.get("model", "gpt-4o-mini")

        if model in self.models:
            client = self.models[model]
            logger.debug(f"llm args:{kwargs}")

            try:
                # 调用openai.chat.completions.create
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,  # 假设session.messages是已正确格式化的消息列表
                    temperature=kwargs.get("temperature", 0.3),
                    stream = kwargs.get("stream",True),
                )
                return response

            except Exception as e:
                # 异常处理逻辑...
                # 根据异常类型进行相应处理
                logger.exception("[ChatGPT] Exception: {}".format(e))
                return {"completion_tokens": 0, "content": e}
