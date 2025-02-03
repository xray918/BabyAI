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

def is_echo(text1, text2, similarity_threshold=0.8):
    """
    检查两段文本是否可能是回声，支持部分匹配和截断情况
    """
    def clean_text(text):
        # 移除所有标点符号和空白字符
        text = re.sub(r'[^\w\s]', '', text)
        # 移除所有空白字符
        text = re.sub(r'\s+', '', text)
        # 统一同音字
        text = text.replace('她', '他').replace('它', '他')
        return text.lower()
    
    def compute_similarity(s1, s2):
        # 如果完全相同，直接返回1.0
        if s1 == s2:
            return 1.0
            
        # 如果其中一个是另一个的子串，返回较高的相似度
        if s1 in s2 or s2 in s1:
            return 0.9
            
        # 滑动窗口匹配
        def sliding_window_match(long_str, short_str, window_size):
            max_similarity = 0
            for i in range(len(long_str) - window_size + 1):
                window = long_str[i:i + window_size]
                # 计算窗口内的字符匹配度
                matches = sum(1 for a, b in zip(window, short_str) if a == b)
                similarity = matches / window_size
                max_similarity = max(max_similarity, similarity)
            return max_similarity
        
        # 计算连续的最长公共子串
        def longest_common_substring(str1, str2):
            m, n = len(str1), len(str2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            max_len = 0
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if str1[i-1] == str2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                        max_len = max(max_len, dp[i][j])
            return max_len
        
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0
            
        # 计算滑动窗口相似度
        window_size = min(len1, len2, 20)  # 使用较小的窗口大小
        if len1 > len2:
            window_sim = sliding_window_match(s1, s2, window_size)
        else:
            window_sim = sliding_window_match(s2, s1, window_size)
            
        # 计算最长公共子串相似度
        substr_len = longest_common_substring(s1, s2)
        substr_sim = substr_len / min(len1, len2)
        
        # 如果有较长的连续匹配或高相似度的滑动窗口匹配，判定为回声
        if substr_sim > 0.6 or window_sim > 0.7:
            return 0.9
            
        return max(substr_sim, window_sim)

    # 清理两段文本
    cleaned_text1 = clean_text(text1)
    cleaned_text2 = clean_text(text2)
    
    # 如果清理后的文本太短，不判断为回声
    if len(cleaned_text1) < 3 or len(cleaned_text2) < 3:
        return False
        
    # 处理部分匹配的情况
    longer_text = cleaned_text1 if len(cleaned_text1) > len(cleaned_text2) else cleaned_text2
    shorter_text = cleaned_text2 if len(cleaned_text1) > len(cleaned_text2) else cleaned_text1
    
    # 对较长文本进行分段检查
    segment_length = len(shorter_text)
    overlap = segment_length // 2  # 50%的重叠
    
    # 1. 检查整体相似度
    if compute_similarity(cleaned_text1, cleaned_text2) >= similarity_threshold:
        return True
        
    # 2. 使用滑动窗口检查部分匹配
    for i in range(0, len(longer_text) - segment_length + 1, overlap):
        segment = longer_text[i:i + segment_length]
        if compute_similarity(segment, shorter_text) >= similarity_threshold:
            return True
            
    # 3. 检查较短文本是否是较长文本的任意连续片段
    if len(shorter_text) >= 10:  # 只对较长的文本进行此检查
        for i in range(len(longer_text) - len(shorter_text) + 1):
            segment = longer_text[i:i + len(shorter_text)]
            if compute_similarity(segment, shorter_text) >= similarity_threshold * 0.8:  # 稍微降低阈值
                return True
    
    return False

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
        recognizer.adjust_for_ambient_noise(source, 2)
        while True:
            try:
                if stream_speak.get_speaking_status():
                    time.sleep(0.1)
                    continue

                print("\n请说...")
                try:
                    if stream_speak.get_speaking_status():
                        continue
                        
                    audio = None
                    start_time = time.time()
                    while time.time() - start_time < 3:
                        try:
                            if stream_speak.get_speaking_status():
                                break
                            audio = recognizer.listen(source, timeout=0.5, phrase_time_limit=10)
                            break
                        except sr.WaitTimeoutError:
                            continue
                    
                    if audio is None or stream_speak.get_speaking_status():
                        continue
                    
                    stream_speak.text_queue.put({"type": "flag", "flag": "listen_over"})
                    
                    question = recognizer.recognize_google(audio, language='zh-CN')
                    
                    # 使用改进的回声检测
                    if messages and messages[-1]["role"] == "assistant":
                        last_response = messages[-1]["content"]
                        # 检查完整匹配
                        if is_echo(question, last_response):
                            continue
                        # 检查部分匹配（前半部分）
                        if len(last_response) > 20:
                            first_half = last_response[:len(last_response)//2]
                            if is_echo(question, first_half):
                                continue
                        # 检查问题是否是回复的前缀
                        if len(question) < len(last_response) and is_echo(question, last_response[:len(question)]):
                            continue
                    
                    print(f"\n用户：{question}")

                    if stream_speak.get_speaking_status():
                        continue

                    messages.append({"role": "user", "content": question})
                    
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

                    if chunk_buf:
                        stream_speak.text_queue.put({"type": "payload", "payload": chunk_buf})
                    
                    stream_speak.text_queue.put({"type": "flag", "flag": "over"})
                    messages.append({"role": "assistant", "content": content})
                    print(f"AI：{content}\n")

                    time.sleep(0.5)
                    
                    recognizer.adjust_for_ambient_noise(source, duration=0.1)
                    try:
                        recognizer.listen(source, timeout=0.1, phrase_time_limit=0.1)
                    except sr.WaitTimeoutError:
                        pass

                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    continue

            except sr.RequestError as e:
                print(f"语音识别服务异常: {e}")
            except Exception as e:
                print(f"发生错误: {e}")
                stream_speak.text_queue.put({"type": "flag", "flag": "over"})

if __name__ == "__main__":
    try:
        print("\n对话开始，请说话...\n")
        listen()
    except KeyboardInterrupt:
        print("\n程序结束。")
    finally:
        StreamSpeak().close()
