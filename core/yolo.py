from ultralytics import YOLO
import cv2
from core.comm import *
from core.ai import *
from collections import Counter
import easyocr
import torch
import asyncio
from collections import Counter
import time
import logging
import warnings
import configparser

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

# 设置环境变量，避免Windows DLL路径警告
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 抑制Windows系统级别的文件查找警告
if sys.platform == 'win32':
    import ctypes
    kernel32 = ctypes.windll.kernel32
    # 设置错误模式，抑制"所提供的模式无法找到文件名"的系统提示
    SEM_FAILCRITICALERRORS = 0x0001
    SEM_NOGPFAULTERRORBOX = 0x0002
    SEM_NOOPENFILEERRORBOX = 0x8000
    kernel32.SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX)

from paddleocr import PaddleOCR

# 全局参数
OCR_READER = PaddleOCR(
    use_angle_cls=True, 
    lang="ch", 
    use_gpu=torch.cuda.is_available(), 
    show_log=False,
    use_mp=False,  # 禁用多进程，避免Windows路径问题
    enable_mkldnn=False  # 禁用MKLDNN加速，避免DLL问题
)
# USE_GPU = torch.cuda.is_available()
# OCR_READER = easyocr.Reader(['ch_sim', 'en'], gpu=USE_GPU)

Card_MODEL = YOLO(r"model\card.pt")
if torch.cuda.is_available():
    Card_MODEL.to('cuda')
else:
    Card_MODEL.to('cpu')

Button_MODEL = YOLO(r"model\button.pt")
if torch.cuda.is_available():
    Button_MODEL.to('cuda')
else:
    Button_MODEL.to('cpu')

# 检查牌
def Yolo_Cards(img):
    confidence_min = 0.5

    labels = set()
    Card_MODEL.conf = confidence_min
    original_image = cv2.imread(img)
    if original_image is None:
        BoxText(f"错误：无法读取图片 {img}")
        return None
    orig_height, orig_width = original_image.shape[:2]
    max_dim = max(orig_height, orig_width)
    imgsz = max(320, min(1280, (max_dim + 31) // 32 * 32))
    results = Card_MODEL.predict(source=img, imgsz=imgsz, save=False, project='temp/yo/', verbose=False)
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            class_index = int(box.cls.item())
            confidence = box.conf.item()
            label = result.names.get(class_index, None).lower()
            # BoxText(fr"日志：置信度={confidence}，标签={label}")
            if confidence >= confidence_min:
                labels.add(label)

    if len(labels) == 0:
        labels = None
    return labels

def Yolo_Button(img, type):
    confidence_min = 0.4
    if type == "fold" or type == "clock" or type == "user":
        confidence_min = 0.1

    all_labels = []  # 使用列表记录所有检测到的标签

    Button_MODEL.conf = confidence_min
    original_image = cv2.imread(img)
    if original_image is None:
        BoxText(f"错误：无法读取图片 {img}")
        if type == "clock":
            return -1, -1
        else:
            return -1

    orig_height, orig_width = original_image.shape[:2]
    max_dim = max(orig_height, orig_width)
    imgsz = max(320, min(1280, (max_dim + 31) // 32 * 32))
    results = Button_MODEL.predict(source=img, imgsz=imgsz, save=False, project='temp/yo/', verbose=False)

    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            class_index = int(box.cls.item())
            confidence = box.conf.item()
            label = result.names.get(class_index, None).lower()
            # BoxText(fr"日志：置信度={confidence}，标签={label}")
            if confidence >= confidence_min:
                all_labels.append(label)

    if not all_labels:  # 如果列表为空，返回 None
        if type == "clock":
            return -6, -6
        else:
            return -2

    label_counts = Counter(all_labels)
    if type == "clock":
        keys = ["b.nz", "b.33", "b.50", "b.75", "b.100", "b.120", "b.zyjz", "b.qp", "b.allyin"]
        clock_count = sum(label_counts.get(key, 0) for key in keys)
        player_count = max(label_counts.get("s.zw", 0), 1)
        return clock_count, player_count
    elif type == "user":
        keys = ["s.bjbss", "s.zdrsl", "s.zss", "b.jzrcl"]
        user_count = sum(label_counts.get(key, 0) for key in keys)
        return user_count
    elif type == "fold":
        qt_count = label_counts.get("s.qp", 0)
        ddz_count = label_counts.get("s.ddz", 0)
        zl_count = label_counts.get("s.zl", 0)
        fold_count = max(qt_count, ddz_count, zl_count)
        return fold_count
    else:
        unique_labels = set(all_labels)
        return unique_labels

def Yolo_Dw(img):
    # 加载和裁剪图像
    image = cv2.imread(img) if isinstance(img, str) else img
    if image is None or (isinstance(img, str) and not os.path.exists(img)): return -1, None, 0, {}
    height, width = image.shape[:2]
    cropped_image = image[int(height * 0.06):-int(height * 0.15), :]
    os.makedirs("temp", exist_ok=True)
    cv2.imwrite("temp/dw_cut.jpg", cropped_image)

    # 划分区域
    height = cropped_image.shape[0]
    row_heights = [height // 5] * 5
    shift_up = int(image.shape[0] * 0.04)
    dealer_conf, dealer_position = 0, None
    position_states, blocks = {}, []
    for row in range(5):
        y_start, y_end = sum(row_heights[:row]), sum(row_heights[:row + 1])
        if row not in (0, 4): y_start, y_end = max(0, y_start - shift_up), min(height, y_end)
        if row in (0, 4):
            blocks.append((y_start, y_end, int(width * 0.3), int(width * 0.3) + int(width * 0.4)))
        else:
            blocks.extend([(y_start, y_end, 0, int(width * 0.35)), (y_start, y_end, width - int(width * 0.35), width)])

    # 定义状态标签并检测
    state_labels = {"s.xz": "下注", "s.jz": "加注", "s.gz": "跟注", "s.rp": "让牌", "s.qp": "弃牌", "s.ddz": "空位", "s.kw": "空位", "s.zl": "空位", "s.allin": "allin"}
    for idx, (y_start, y_end, x_start, x_end) in enumerate(blocks):
        block = cropped_image[y_start:y_end, x_start:x_end]
        cv2.imwrite(f"temp/user_{idx}.jpg", block)
        results = Button_MODEL.predict(source=f"temp/user_{idx}.jpg", imgsz=round(block.shape[1] // 2 // 32) * 32, save=False, project='temp/detect', name='识别结果', verbose=False)
        user_labels = [(Button_MODEL.names[int(box.cls[0])], float(box.conf[0])) for result in results for box in result.boxes.cpu().numpy()]
        for label, conf in user_labels:
            if label == "s.zj" and conf > dealer_conf: dealer_conf, dealer_position = conf, idx
        for label, conf in user_labels:
            if label in state_labels: position_states[idx] = {"state": state_labels[label], "conf": conf}; break
        else:
            position_states[idx] = {"state": None, "conf": 0}

    # 映射玩家位置并计算座位
    player_map = {0: 4, 1: 3, 2: 5, 3: 2, 4: 6, 5: 1, 6: 7, 7: 0}
    dealer_pos_num = player_map.get(dealer_position, -1)
    positions = {0: "庄家", 1: "小盲", 2: "大盲", 3: "枪口", 4: "枪口+1", 5: "中间", 6: "截止", 7: "劫"}
    has_bet_or_raise = any(position_states.get([k for k, v in player_map.items() if v == pos][0], {"state": None})["state"] in ["下注", "加注"] for pos in range(8))
    after = sum(1 for pos in range(1, 8) for user_idx in [[k for k, v in player_map.items() if v == pos][0]] if
                position_states.get(user_idx, {"state": None})["state"] is None or (has_bet_or_raise and position_states.get(user_idx, {"state": None})["state"] == "让牌"))
    seat_positions = {pos: "未知" for pos in range(8)} if dealer_pos_num == -1 else {pos: positions.get((pos - dealer_pos_num) % 8, "未知") for pos in range(8)}

    # 确定下注者和加注者
    first_bettor, raiser = None, None
    for pos in range(8):
        user_idx = [k for k, v in player_map.items() if v == pos][0]
        state = position_states.get(user_idx, {"state": None})["state"]
        if state == "下注" and first_bettor is None and raiser is None:
            first_bettor = pos
        elif state == "加注":
            raiser = pos

    # 构建历史记录
    history = "本次行动处在_circle_，玩家状态："
    for pos in range(8):
        user_idx = [k for k, v in player_map.items() if v == pos][0]
        state = position_states.get(user_idx, {"state": None})["state"]
        state = "我自己，正在行动" if pos == 0 else ("等待行动" if state is None else state)
        if pos == raiser and state == "加注":
            state = "加注_bett_"
        elif pos == first_bettor and state == "下注" and raiser is None:
            state = "下注_bett_"
        history += f"{seat_positions[pos]}位（{state}）{'、' if pos != 7 else '。'}"

    return history, after, seat_positions[0] + '位'

async def Ocr_Nmuber(img, ocr_type):
    """
    🚀 性能优化：直接使用AI识别数字，跳过OCR步骤
    原因：OCR识别率低且耗时，直接使用AI更快更准
    """
    if ocr_type in ("bett", "chip", "hold"):
        rule = "帮我找出这张图片上的数字，如果有多个返回比较大的这个数字，不要回车符等，必须只返回一个纯净数字。"
        renum = Ali_Number(img, rule).strip()
        # 解析AI返回的JSON格式
        try:
            import json
            ai_result = json.loads(renum)
            # 尝试从不同的key中获取数值
            renum = str(ai_result.get('answer') or ai_result.get('number') or ai_result.get('result') or '0')
        except (json.JSONDecodeError, AttributeError, TypeError):
            # 如果不是JSON或解析失败，保持原值（可能已经是纯数字）
            pass
        return renum
    else:
        return "0.1"  # 盲注默认值

# ============================================================================
# 原OCR版本的Ocr_Nmuber函数（已废弃，保留以备后用）
# 如需恢复OCR识别，取消注释以下代码并替换当前的Ocr_Nmuber函数
# ============================================================================
"""
原OCR识别版本（已禁用）：

先尝试PaddleOCR识别，失败后调用AI识别
缺点：OCR识别率低，每次失败后都要重新调用AI，浪费时间

async def Ocr_Nmuber_OCR_Version(img, ocr_type):
    result = OCR_READER.ocr(img, cls=True)
    if not result or not isinstance(result, list) or not result[0]:
        if ocr_type == "bett" or ocr_type == "chip" or ocr_type == "hold":
            rule = "帮我找出这张图片上的数字，如果有多个返回比较大的这个数字，不要回车符等，必须只返回一个纯净数字。"
            renum = Ali_Number(img, rule).strip()
            BoxText(f"错误：--Yolo_Ocr识别{img}失败，改用AI识别={renum}")
            try:
                import json
                ai_result = json.loads(renum)
                renum = str(ai_result.get('answer') or ai_result.get('number') or ai_result.get('result') or '0')
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass
        else:
            renum = "0.1"
        return renum

    detections, numeric_results = result[0], []
    for detection in detections:
        text, con = detection[1]
        text = text.replace("$", "").replace("￥", "").replace("（", "(").replace("）", ")").replace("：", ":")
        text = str(text).lower()

        if ocr_type in ("chip", "hold"):
            numeric_results.append(float(text) if text.replace('.', '', 1).isdigit() else 0)
        elif ocr_type == "bett":
            if any(kw in text for kw in ("s", "让", "牌")):
                return -123.456
            elif any(kw in text for kw in ("allin", "all", "in")):
                return 999.999
            else:
                numeric_results.append(float(text) if text.replace('.', '', 1).isdigit() else -234.567)
        elif ocr_type == "blind" and "注" in text and "/" in text:
            try:
                blind = float(text.split("注:")[1].split("(")[0].strip().split("/")[-1])
                if "钻石" in text:
                    blind = blind * 10
                return blind
            except Exception:
                return 0.0123

    return max(numeric_results) if numeric_results else 0.123

"""每10分钟清理temp文件夹下所有子文件夹中的文件，但保留子文件夹"""

def Clean_Temp():
    while True:
        try:
            temp_dir = 'temp'
            if os.path.exists(temp_dir):
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        os.remove(file_path)
                        BoxText(f"清理日志：已删除文件 {file_path}")
                BoxText(f"清理日志：temp文件夹清理完成 | 时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            else:
                BoxText("清理日志：temp文件夹不存在，跳过清理")
        except Exception as e:
            BoxText(f"清理错误: {str(e)}")
        time.sleep(600)  # 每10分钟（600秒）执行一次
