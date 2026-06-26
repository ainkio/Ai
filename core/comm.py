# 打包路径
import os
import sys
import datetime
import colorama
from colorama import Fore, Back, Style, init
import base64
from PIL import Image
import time
from functools import lru_cache
import uuid
import hashlib
import json
import psutil
import torch
import configparser
import random

start_time = None
colorama.init()
init(autoreset=True)  # 初始化colorama，autoreset=True会在每次打印后自动重置颜色

# 从集中式配置模块导入所有配置
from core.config import *

global_BoxText_count = 0
global_BoxText_xcount = 0
last_was_dot = False

color_map = {'green': Fore.GREEN, 'red': Fore.RED, 'white': Fore.WHITE, 'blue': Fore.BLUE, 'black': Fore.BLACK, 'yellow': Fore.YELLOW, 'magenta': Fore.MAGENTA, 'cyan': Fore.CYAN}
def BoxText(text):
    global global_BoxText_count, global_BoxText_xcount, last_was_dot
    current_time = datetime.datetime.now().strftime('%H:%M:%S')

    # Available colors for random selection
    available_colors = ['WHITE', 'GREEN', 'RED', 'YELLOW', 'CYAN', 'MAGENTA', 'LIGHTBLACK_EX']

    # 根据文本内容设定颜色
    if text.startswith('日志：'):
        color = 'WHITE'
    elif text.startswith('成功：'):
        color = 'GREEN'
    elif text.startswith('错误：'):
        color = 'RED'
    elif text.startswith('警告：'):
        color = 'RED'
    elif text.startswith('调试：'):
        color = 'GREEN'
    elif text.startswith('重要：'):
        color = 'YELLOW'
    elif text.startswith('消息：'):
        color = 'CYAN'
    elif text.startswith('通知：'):
        color = 'MAGENTA'
    elif text.startswith('输出：'):
        color = 'LIGHTBLACK_EX'
    else:
        color = 'WHITE'

    color_code = color_map.get(color.lower(), Fore.WHITE)
    global_BoxText_count += 1

    if text == "." or len(text) <= 2:
        # Random color for dots
        random_color = random.choice(available_colors)
        random_color_code = color_map.get(random_color.lower(), Fore.WHITE)

        if global_BoxText_xcount == 66:
            print(f"{random_color_code}{text}")
            global_BoxText_xcount = 0
        else:
            global_BoxText_xcount += 1
            print(f"{random_color_code}{text}", end="")
        last_was_dot = True
    else:
        # If previous output was a dot, print a newline first
        if last_was_dot:
            print()
            global_BoxText_xcount = 0
        print(f"{color_code} {global_BoxText_count} [{current_time}] {text}")
        last_was_dot = False


# 输出文字模块


# 连接mysql
import mysql.connector
from mysql.connector import Error


def Mysql_Link():
    """建立与MySQL数据库的连接"""
    try:
        connection = mysql.connector.connect(host=DB_HOST,  # 数据库主机地址
                                             user=DB_USER,  # 数据库用户名
                                             password=DB_PASSWORD,  # 数据库密码
                                             database=DB_NAME,  # 数据库名
                                             charset='utf8mb4'  # 兼容emoji
                                             )
        if connection.is_connected():
            # BoxText(f"日志：core.comm.Mysql_Link.成功")
            return connection
    except Error as e:
        BoxText(f"错误：core.comm.Mysql_Link.错误.{e}")
        return None


def Mysql_Row(sql):
    connection = Mysql_Link()
    if not connection:
        BoxText(f"日志：core.comm.Mysql_Link.No database connection established.")
        return

    try:
        cursor = connection.cursor()
        query = sql
        cursor.execute(query)

        # 判断是SELECT、UPDATE还是INSERT
        if sql.strip().upper().startswith("SELECT"):
            # 处理SELECT语句
            column_names = [desc[0] for desc in cursor.description]
            records = cursor.fetchall()
            result = [dict(zip(column_names, row)) for row in records]
        elif sql.strip().upper().startswith("UPDATE") or sql.strip().upper().startswith("INSERT"):
            # 处理UPDATE和INSERT语句
            connection.commit()  # 提交事务
            # 检查是否为插入操作，并返回最后插入的ID
            if sql.strip().upper().startswith("INSERT"):
                result = cursor.lastrowid  # 返回最后插入的ID
            else:
                result = cursor.rowcount  # 返回受影响的行数

        else:
            # 其他类型的SQL语句
            connection.commit()  # 提交事务
            result = cursor.rowcount  # 返回受影响的行数

        return result

    except Error as e:
        BoxText(f"错误：core.comm.Mysql_Link.Error while executing SQL. {e}")
        # BoxText(f"消息：core.comm.Mysql_Link.SQL={sql}")
        return None

    finally:
        cursor.close()
        connection.close()


# 连接mysql


# 当前文件路径
def Abs_Path():
    abs_path = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.dirname(abs_path) + "/"
    return abs_path


# 当前文件路径

# 输出详细错误
import sys
import traceback


def Error_Detail(e):
    exc_type, exc_value, exc_traceback = sys.exc_info()
    file_name = exc_traceback.tb_frame.f_code.co_filename
    line_number = exc_traceback.tb_lineno
    BoxText(f"错误：产生错误{e}，错误发生在文件 {file_name} 的第 {line_number} 行。")
    traceback.print_exc()


# 输出详细错误

# 唯一标识
@lru_cache(maxsize=1)  # 只缓存最新的结果
def Uuid():
    import platform
    cpu = f"{psutil.cpu_count(logical=False)}|{psutil.cpu_count(logical=True)}|{psutil.cpu_freq().max:.2f}|{platform.processor()}"
    sha_signature = hashlib.sha256(cpu.encode()).hexdigest()
    return sha_signature


def Check_Auth():
    import requests
    uuid = Uuid()
    # 从配置读取授权URL，未配置时以本地模式运行
    url = POKER_AUTH_URL if POKER_AUTH_URL else f'{GLOBAL_DOMAIN}/auth/'
    payload = {"uuid": uuid}
    try:
        response = requests.post(url, data=payload)
        response_data = response.json()
        if response_data.get("status") == "ok":
            BoxText(f"重要：{response_data.get('message', '授权成功')}")
            return True
        else:
            input(fr"机器码【{uuid}】暂无授权，请先授权......")
            return False

    except Exception as e:
        # 未配置授权服务器时，允许本地运行（开源模式）
        print(f"提示：未连接授权服务器，以本地模式运行: {e}")
        return True


# 唯一标识

# 计算耗时
def Time_Count(action):
    global start_time
    if action == 'start':
        start_time = time.time()
        BoxText("计时开始...")
    elif action == 'end':
        if start_time is None:
            BoxText("请先调用 'start' 开始计时")
        else:
            elapsed_time = time.time() - start_time
            BoxText(f"耗时: {elapsed_time:.2f} 秒")
            return elapsed_time


# 计算耗时

# 图片缩放
def Img_Resize(img, max):
    with Image.open(img) as img_obj:
        width, height = img_obj.size
        if width > max:
            new_height = int(max * height / width)
            resized_img = img_obj.resize((max, new_height), Image.LANCZOS)
            resized_img.save(img, format='JPEG', quality=95)


# 图片缩放

# 图片转base64
def Img_Base64(img):
    with open(img, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string  # 图片转base64


# 转义
def Str_Escape(str):
    modified_str = str.replace("\\", "~").replace("`", "~").replace("'", "~")
    return modified_str


# 字符串转JSON
def Str_Json(str):
    start = str.find('{')
    end = str.rfind('}') + 1
    if start != -1 and end != -1:
        valid_json_str = str[start:end]
    else:
        start = str.find('[')
        end = str.rfind(']') + 1
        if start != -1 and end != -1:
            valid_json_str = str[start:end]

    try:
        json_obj = json.loads(valid_json_str)
    except Exception as e:
        json_obj = []
    return json_obj


# 播放声音
import win32com.client
from threading import Thread
import pythoncom
# asyncio.run(Play_Audio(action))
async def Play_Audio(text):
    def speak_thread(text):
        pythoncom.CoInitialize()
        try:
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Volume = 100
            speaker.Speak(text)
        finally:
            pythoncom.CoUninitialize()
    thread = Thread(target=speak_thread, args=(text,))
    thread.start()

# 管理员运行
def Sys_Power():
    import ctypes
    def is_admin():
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False

    if not is_admin():
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
        sys.exit()


# 读取配置文件（委托给集中式配置模块 core/config.py）
def Read_Config(type):
    from core.config import Read_Config as _read_config
    return _read_config(type)