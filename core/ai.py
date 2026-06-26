import os
from openai import OpenAI, AsyncOpenAI
from core.comm import *
from core.config import *
from openai import APITimeoutError
import shutil
from datetime import datetime
from treys import Card, Evaluator, Deck
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import re

# ***** 客户端初始化（密钥从 core/config.py 统一读取）*****
Ali_Client = OpenAI(api_key=ALI_API_KEY, base_url=ALI_BASE_URL)
Ali_Client_Asyn = AsyncOpenAI(api_key=ALI_API_KEY, base_url=ALI_BASE_URL, timeout=10)

DeepSeek_Client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

Xai_Client = OpenAI(api_key=XAI_API_KEY, base_url=XAI_BASE_URL, timeout=10)


# ***** 阿里识图 *****
# 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
def Ali_Number(vimg, rule, model='qwen3-vl-flash'):
    BoxText(f"消息：Ali_Number正在分析数字，图片{vimg}")
    ALI_ID = '你是数字识别专家。'
    ALI_QUESTION = rule
    if not vimg.startswith("http"):
        vimg = f"data:image/png;base64,{Img_Base64(img=vimg)}"
    try:
        completion = Ali_Client.chat.completions.create(model=model, messages=[{"role": "system", "content": [{"type": "text", "text": ALI_ID}]}, {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"{vimg}"}}, {"type": "text", "text": ALI_QUESTION}]}], timeout=3, response_format={'type': 'json_object'})
        return completion.choices[0].message.content
    except Exception as e:
        BoxText(f"警告：Ali_Number，图片{vimg}，异常={str(e)}")
        return "[]"

async def Ali_Cards_Async(vimg, model='qwen3-vl-plus'):
    ALI_ID = '你是德州扑克游戏和图片识别的专家，你能准确的识别出图片上的德州扑克决策所需要的信息，尤其要仔细识别扑克牌花色。'
    ALI_QUESTION = """都有什么扑克牌？用点数+（黑桃s、红桃h、梅花c、方块d）表示花色，点数10用T表示，用JSON格式返回给我，例如：["As","Kd"]"""
    if not vimg.startswith("http"):
        vimg = f"data:image/png;base64,{Img_Base64(img=vimg)}"
    try:
        completion = await Ali_Client_Asyn.chat.completions.create(model=model, messages=[{"role": "system", "content": [{"type": "text", "text": ALI_ID}]}, {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"{vimg}"}}, {"type": "text", "text": ALI_QUESTION}]}], timeout=10, response_format={'type': 'json_object'})
        return completion.choices[0].message.content
    except Exception as e:
        BoxText(f"警告：Ali_Cards_Async超时或连接错误，图片{vimg}，异常={str(e)}")
        return "[]"

# ***** 智能决策体 *****
amd, dmd, xmd, aipa, iden, rule, prefilter_enabled, correction_enabled = Read_Config(type="mode")

def Good_Idea(describe, cnum):
    global response
    start_time = time.time()
    ap = aipa[cnum] if 0 <= cnum < len(aipa) else aipa
    response = {'type': 'json_object'}

    if ap == "a":
        client = Ali_Client
        model = amd
    elif ap == "d":
        client = DeepSeek_Client
        model = dmd
    elif ap == "x":
        client = Xai_Client
        model = xmd

    identity = iden
    describe = describe.replace("999.999", "allin").replace("-123.456", "0")
    question = f"{describe}\n"

    BoxText(f"通知：最终提问 = {question.replace("\n", "")}")
    messages = [{"role": "system", "content": f"{identity}"}, {"role": "user", "content": f"{question}"}, ]

    try:
        completion = client.chat.completions.create(model=model, messages=messages, temperature=0, response_format=response, stream=False)
        result = completion.choices[0].message.content
        # 非标准JSON格式时，提炼标准JSON格式
        if "json" in result.lower():
            pattern = r'"action"\s*:\s*"([^"]*)"\s*,\s*"betting"\s*:\s*"([^"]*)"'
            match = re.search(pattern, result)
            if match:
                action_value = match.group(1)
                betting_value = match.group(2)
                result = f'{{"action": "{action_value}","betting": "{betting_value}"}}'

        end_time = time.time()
        elapsed_time = end_time - start_time
        BoxText(f"成功：决策{ap} | 耗{elapsed_time:.2f}秒 = {result.replace("\n", "")}")

        # 写入文件
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('x_poker.txt', 'a', encoding='utf-8') as f:
            f.write(f"*****{current_time}*****\n")
            f.write(f"{identity}\n\n")
            f.write(f"{question}\n")
            f.write(f"答案：{result}\n\n")

    except APITimeoutError as e:

        BoxText("错误: 请检查网络联接和开启代理。")
        return {'action': 'error', 'betting': '请检查网络联接和开启代理'}
    except Exception as e:
        BoxText(f"发生了一个错误: {e}")
        return {'action': 'error', 'betting': '发生了未知错误，请稍后再试'}
    return result

# ***** 智能决策体 *****

if __name__ == "__main__":
    qu = """你的手牌：['6s', '6d']，公牌：['9c', '7h', '2s']，剩余对手：5，锅底筹码：18.6，你的筹码：18.5，当前需下：9.3，你的位置：中间位，游戏阶段：翻牌圈，在你后面行动玩家数：4个。
    本手游戏行动历史：
    第1次行动：本次行动处在底牌圈，玩家状态：中间位（你自己，call 1.2）、截止位（等待行动）、劫位（等待行动）、庄家位（等待行动）、小盲位（等待行动）、大盲位（等待行动）、枪口位（加注1.2）、枪口+1位（跟注）。
    第2次行动，本次行动处在翻牌圈，玩家状态：中间位（你自己，还未行动）、截止位（等待行动）、劫位（等待行动）、庄家位（等待行动）、小盲位（让牌）、大盲位（弃牌）、枪口位（下注9.3）、枪口+1位（弃牌）。
    """
    gi = Good_Idea(describe=qu)
