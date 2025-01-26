import re
import queue
import threading
import time

from llm import *
from speak import *



# 用于控制程序状态
# is_listening = True
# is_speaking = False


def extract_all_sentences(text):
    """
    提取字符串中所有完整的句子（以句号、感叹号或问号结尾），
    同时返回剩余的部分。
    """
    pattern = re.compile(r'([^。！？.!?]*[。！？.!?])')
    sentences = pattern.findall(text)

    last_match = pattern.search(text)
    while last_match:
        end_pos = last_match.end()
        last_match = pattern.search(text, last_match.end())

    remaining_text = text[end_pos:] if 'end_pos' in locals() else text
    sentences_str = ''.join(sentences)

    return sentences_str, remaining_text

def listen():
    """监听用户的语音输入"""
    global is_listening, is_speaking
    messages = [
        {
            "role": "system",
            "content": """你是一个四岁的中国小女孩，名字叫米娅，你能用最简单的小孩子的语言，调动其他小朋友和你对话的兴趣,因为文字要通过TTS要转成语音对话，所以文字要口语化，不要包含难以转化为口语的符合，
            例如：
            我：怎么养小狗狗
            你：要定时地喂它，带它遛弯，给他洗澡等等
            我：你喜欢玩啥
            你：我喜欢搭乐高，我用乐高搭过一个长长的火箭哦，你喜欢啥？
            我：我喜欢玩游戏，特别是捉迷藏"""
                    },
    ]
    stream_speak = StreamSpeak()
    llm = LLM()

    with sr.Microphone() as source:
        # print(f"调整噪音水平，不要说话...")
        recognizer.adjust_for_ambient_noise(source, 2)
        while True:
            try:
                # 检查AI是否正在说话
                if stream_speak.get_speaking_status():
                    time.sleep(0.2)  # 在树莓派上增加等待时间
                    continue

                print("请说...")
                try:
                    # 在录音前再次确认AI没有在说话
                    if stream_speak.get_speaking_status():
                        continue
                    
                    # 设置较短的超时时间
                    audio = recognizer.listen(source, timeout=3, phrase_time_limit=10)
                    
                    # 录音后立即检查AI是否开始说话
                    if stream_speak.get_speaking_status():
                        print("检测到AI正在说话，丢弃当前录音")
                        continue
                    
                    # 标记录音结束
                    stream_speak.text_queue.put({"type": "flag", "flag": "listen_over"})
                    
                    # 等待一小段时间确保标记被处理
                    time.sleep(0.1)
                    
                    print("识别...")
                    question = recognizer.recognize_google(audio, language='zh-CN')
                    print(f"you: {question}")

                    # 在发送到LLM之前确保AI没有在说话
                    if stream_speak.get_speaking_status():
                        print("AI正在说话，跳过这次输入")
                        continue

                    messages.append({"role": "user", "content": question})
                    
                    # 设置说话状态
                    stream_speak.set_speaking_active(True)
                    stream_speak.text_queue.put({"type": "flag", "flag": "speak_start"})
                    
                    # 获取AI响应
                    response = llm.reply_text_stream(messages, model="gpt-4o-mini", stream=True, temperature=0.7)
                    remaining_text = ""
                    chunk_buf = ""
                    content = ""

                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            chunk_buf += chunk.choices[0].delta.content
                            content += chunk.choices[0].delta.content
                            if len(chunk_buf) > 100:
                                sentence, remaining_text = extract_all_sentences(chunk_buf)
                                if sentence:
                                    chunk_buf = remaining_text
                                    stream_speak.text_queue.put({"type": "payload", "payload": sentence})

                    # 处理最后的文本
                    if chunk_buf:
                        stream_speak.text_queue.put({"type": "payload", "payload": chunk_buf})
                    
                    # 标记AI说话结束
                    stream_speak.text_queue.put({"type": "flag", "flag": "over"})
                    messages.append({"role": "assistant", "content": content})
                    print(f"AI：{content}")

                    # 等待一小段时间确保音频播放完成
                    time.sleep(0.2)

                except sr.WaitTimeoutError:
                    print("长时间没有检测到声音，超时。")
                    continue

            except sr.UnknownValueError:
                print("抱歉，我没有听清楚。")
                stream_speak.text_queue.put({"type": "flag", "flag": "over"})
            except sr.RequestError as e:
                print(f"无法连接到语音识别服务; {e}")
            except ValueError as e:
                print(e)
                stream_speak.text_queue.put({"type": "flag", "flag": "over"})
            except Exception as e:
                print(e)
                stream_speak.text_queue.put({"type": "flag", "flag": "over"})



if __name__ == "__main__":
    try:
        listen()
    except KeyboardInterrupt:
        print("程序终止中...")
    finally:
        print("释放媒体资源...")
        StreamSpeak().close()
        print("释放完毕，程序退出。")
