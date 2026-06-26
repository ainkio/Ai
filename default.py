# 在所有导入之前设置Windows错误模式，抑制系统级别的文件查找警告
import sys
if sys.platform == 'win32':
    import ctypes
    kernel32 = ctypes.windll.kernel32
    # 设置错误模式，抑制"所提供的模式无法找到文件名"的系统提示
    SEM_FAILCRITICALERRORS = 0x0001
    SEM_NOGPFAULTERRORBOX = 0x0002
    SEM_NOOPENFILEERRORBOX = 0x8000
    kernel32.SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX)

from core.comm import *
from core.ai import *
from core.cal import *
from core.yolo import *
import win32process, win32api, win32con, numpy as np, pyautogui, win32gui, time, os
import tkinter as tk
import asyncio, threading, cv2
from concurrent.futures import ThreadPoolExecutor
import warnings
import json
import os

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")

# 全局常量
DEFAULT_BG_COLOR = '#2A2D2E'
WIN_AFTER = 50  # 窗口更新间隔 (ms)
WIN_SLEEP = 0.3  # 窗口查找间隔 (s)
TIMEOUT_THRESHOLD = 15  # 超时阈值（秒）
LAYER_BG_DEFAULT = '#001d26'
LAYER_BG_RAISE_CALL = '#0884db'
LAYER_BG_FOLD = '#e63d3a'
SHOW_LEFT_WINDOW = True

# 全局变量
Layer_Bg = LAYER_BG_DEFAULT
gui = None  # 全局 GUI 对象

# 全局状态类
class PokerState:
    def __init__(self):
        self.reset()

    def reset(self, exclude=""):
        self.yo_hand = None
        self.yo_public = None
        self.last_hash = None
        self.consistent_count = 0
        self.hash_consistent_count = 0
        self.last_decision_time = 0
        self.history = []  # 清空历史记录
        self.action_count = 0  # 重置行动次数
        self.is_deciding = False
        self.last_yolo_triggered = {'yo_hand': [], 'yo_public': []} if "|last_yolo_triggered|" not in exclude else self.last_yolo_triggered
        self.last_calc_completed = False
        self.last_mon_chip = 0
        self.last_mon_bett = 0
        self.last_mon_blind = 0
        self.last_mon_hold = 0
        self.mon_hold = 0
        self.last_action = None
        self.last_betting = None
        self.last_out_text = None
        self.no_clock_count = 0
        self.last_update_time = time.time()
        self.ai_hand = None if "|ai_hand|" not in exclude else self.ai_hand
        self.ai_public = []

        # 如果是程序初始化，确保 last_ai_triggered 为 None
        if not hasattr(self, 'last_ai_triggered'):
            self.last_ai_triggered = None

state = PokerState()

def Win_Handle(process_name):
    def callback(wh_hwnd, wh_hwnds):
        try:
            _, pid = win32process.GetWindowThreadProcessId(wh_hwnd)
            h_process = win32api.OpenProcess(win32con.PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            wh_exe_name = win32process.GetModuleFileNameEx(h_process, 0).lower()
            if process_name.lower() in wh_exe_name and win32gui.IsWindowVisible(wh_hwnd):
                if win32gui.GetWindowPlacement(wh_hwnd)[1] != win32con.SW_SHOWMINIMIZED:
                    wh_hwnds.append(wh_hwnd)
        except Exception:
            pass
        return True

    wh_hwnds = []
    win32gui.EnumWindows(callback, wh_hwnds)
    return wh_hwnds[0] if wh_hwnds else None

class PokerGUI:
    def __init__(self, root, hwnd):
        self.root = root
        self.hwnd = hwnd
        self.setup_ui()

    def setup_ui(self):
        self.root.title("智能扑克助手")
        self.root.overrideredirect(True)
        self.root.configure(bg=DEFAULT_BG_COLOR)
        self.root.attributes('-topmost', True)

        if SHOW_LEFT_WINDOW:
            self.main_frame = tk.Frame(self.root, bg=DEFAULT_BG_COLOR)
            self.main_frame.pack(fill="both", expand=True, padx=0, pady=0)
            self.main_frame.rowconfigure(0, weight=1)
            self.main_frame.columnconfigure(0, weight=1)
            self.create_decision_display()

        self.create_top_layer()

    def create_decision_display(self):
        self.decision_label = tk.Label(self.main_frame, text="\n弃牌/等待中.\n", font=("微软雅黑", 19, "bold"), wraplength=450, justify="left", anchor="nw", fg="#F0F0F0", bg=DEFAULT_BG_COLOR)
        self.decision_label.grid(row=0, column=0, sticky="nsew", pady=2, padx=25)

    def create_top_layer(self):
        global Layer_Bg
        self.top_window = tk.Toplevel(self.root)
        self.top_window.overrideredirect(True)
        self.top_window.configure(bg=Layer_Bg)
        self.top_window.attributes('-alpha', 0.7)
        self.top_window.attributes('-topmost', True)
        self.action_label = tk.Label(self.top_window, text="行动：欢迎开局", font=("微软雅黑", 14), fg="white", bg=Layer_Bg)
        self.action_label.pack(expand=True, padx=10)

    def update_layout(self):
        try:
            window_rect = win32gui.GetWindowRect(self.hwnd)
            total_height = window_rect[3] - window_rect[1] - 20
            if SHOW_LEFT_WINDOW:
                main_height = int(total_height * 0.77)
                self.root.geometry(f"500x{main_height}+{window_rect[0] - 510}+{window_rect[1] + 44}")
                self.root.attributes('-alpha', 0.8)
            else:
                self.root.geometry("1x1+0+0")
                self.root.attributes('-alpha', 0)

            top_height = int(total_height * 0.03)
            main_width = window_rect[2] - window_rect[0]
            top_width = int(main_width * 0.956)
            top_x = window_rect[0] + int(main_width * 0.022)
            top_y = window_rect[3] - top_height - 2
            self.top_window.geometry(f"{top_width}x{top_height}+{top_x}+{top_y}")
            self.top_window.configure(bg=Layer_Bg)
            self.action_label.configure(bg=Layer_Bg)
        except Exception as e:
            BoxText(f"布局错误: {str(e)}")

class PokerDetector:
    def __init__(self, hwnd):
        self.hwnd = hwnd
        self.running = True
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self.start_async_loop, daemon=True)
        self.loop_thread.start()
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.detect_thread = threading.Thread(target=self.detect_loop, daemon=True)
        self.detect_thread.start()

    def start_async_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run_async(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    async def card_ai(self, hand_img, public_img):
        if public_img is not None:
            res1, res2 = await asyncio.gather(Ali_Cards_Async(vimg=hand_img), Ali_Cards_Async(vimg=public_img))
        else:
            res1 = await Ali_Cards_Async(vimg=hand_img)
            res2 = "[]"
        result = (Str_Json(res1), Str_Json(res2))
        return result

    async def ocr_tasks(self, screenshot, width, height):
        tasks = []
        regions = {'chip': (int(width * 0.3), int(height * 0.24), width * (1 - 0.3), int(height * 0.35)),
                   'bett': (int(width * 0.62), int(height * 0.75), int(width * 0.87), int(height * 0.87)),
                   'hold': (int(width * 0.35), int(height * 0.927), int(width * 0.65), int(height * 0.98)),
                   'blind': (int(width * 0.3), int(height * 0.55), width * (1 - 0.3), int(height * 0.63))}

        for key, (left, top, right, bottom) in regions.items():
            region = screenshot.crop((left, top, right, bottom))
            img_path = f'temp/mon_{key}.jpg'
            region.save(img_path, quality=95)
            tasks.append(asyncio.create_task(Ocr_Nmuber(img_path, key)))

        results = await asyncio.gather(*tasks)
        ocr_result = dict(zip(regions.keys(), results))
        return ocr_result

    def get_image_hash(self, image):
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
        binary = (resized > resized.mean()).astype(np.uint8)
        return sum([2 ** i for (i, v) in enumerate(binary.flatten()) if v])

    def detect_loop(self):
        while self.running:
            try:
                screenshot, width, height = self.capture_screen()
                asyncio.run(process_poker_state(self, gui, screenshot, width, height))
            except Exception as e:
                BoxText(f"日志：检测线程异常 | {Error_Detail(e)}")
                if SHOW_LEFT_WINDOW:
                    gui.decision_label.config(text="\n异常重置中...\n")
                gui.action_label.config(text="行动：异常重置")
                state.last_update_time = time.time()
            time.sleep(0.1)

    def capture_screen(self):
        left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
        x, y = win32gui.ClientToScreen(self.hwnd, (left, top))
        width, height = right - left, bottom - top
        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        return screenshot, width, height

    def run_yolo_detection(self, screenshot, width, height):
        card_results = self.get_card_regions(screenshot, width, height)
        screenshot.save("temp/real_img.png", quality=95)
        yolo_clock, yolo_player = Yolo_Button(img="temp/real_img.png", type="clock")
        fold_result = self.check_fold(screenshot, width, height)
        yolo_dw_result = Yolo_Dw("temp/real_img.png")
        history, after, my_position = yolo_dw_result if isinstance(yolo_dw_result, (tuple, list)) and len(yolo_dw_result) == 3 else ("", 0, "未知位")
        return {'cards': card_results, 'clock': yolo_clock, 'player': yolo_player, 'fold': fold_result, 'history': history, 'after': after, 'my_position': my_position}

    def run_ocr_tasks(self, screenshot, width, height):
        ocr_results = self.run_async(self.ocr_tasks(screenshot, width, height))
        return ocr_results if ocr_results is not None else {'chip': 0, 'bett': 0, 'hold': 0, 'blind': 0}

    def run_ai_analysis(self, screenshot):
        try:
            screenshot.crop((int(screenshot.width * 0.35), screenshot.height - int(screenshot.height * 0.2), int(screenshot.width * 0.65), screenshot.height - int(screenshot.height * 0.098))).save(
                'temp/hand_img.png', quality=95)
            screenshot.crop((int(screenshot.width * 0.19), int(screenshot.height * (1 - 0.528)), int(screenshot.width * 0.8), int(screenshot.height * (1 - 0.435)))).save('temp/public_img.png',
                                                                                                                                                                          quality=95)
            public_img = 'temp/public_img.png' if os.path.exists('temp/public_img.png') else []
            ai_hand, ai_public = self.run_async(self.card_ai(hand_img='temp/hand_img.png', public_img=public_img))
        except Exception as e:
            ai_hand, ai_public = state.ai_hand, state.ai_public
        return {'hand': ai_hand, 'public': ai_public}

    def check_fold(self, screenshot, width, height):
        fold_region = screenshot.crop((int(width * 0.35), height - int(height * 0.3), int(width * 0.65), height - int(height * 0.148)))
        fold_region.save('temp/fold_img.png', quality=95)
        fold = Yolo_Button(img='temp/fold_img.png', type='fold')
        return int(fold) > 0 if fold is not None else False

    def get_card_regions(self, screenshot, width, height):
        hand_region = screenshot.crop((int(width * 0.35), height - int(height * 0.2), int(width * 0.65), height - int(height * 0.098)))
        hand_region.save('temp/hand_img.png', quality=95)
        public_region = screenshot.crop((int(width * 0.19), int(height * (1 - 0.528)), int(width * 0.8), int(height * (1 - 0.435))))
        public_region.save('temp/public_img.png', quality=95)
        return {'yo_hand': Yolo_Cards('temp/hand_img.png') or [], 'yo_public': Yolo_Cards('temp/public_img.png') or []}

    def normalize_cards(self, cards):
        if cards is None:
            return []
        try:
            return sorted([card.lower() for card in cards])
        except TypeError:
            return []

    def stop(self):
        self.running = False
        self.executor.shutdown(wait=False)
        self.loop.stop()

async def process_poker_state(detector, gui, screenshot, width, height):
    global Layer_Bg
    current_time = time.time()
    state.last_update_time = current_time  # 确保更新

    # 步骤 1：检查截图哈希值是否变化
    current_hash = detector.get_image_hash(screenshot)
    if state.last_hash == current_hash:
        state.hash_consistent_count += 1
        if state.hash_consistent_count < 1:
            return False
    else:
        state.hash_consistent_count = 0
        state.last_hash = current_hash
        #BoxText(f"日志：哈希值变化...")
        BoxText(".")

    # 步骤 2：检查是否已弃牌
    fold_result = detector.check_fold(screenshot, width, height)
    if fold_result:
        state.reset()
        state.no_clock_count = 0
        if SHOW_LEFT_WINDOW:
            gui.decision_label.config(text="\n弃牌/等待中.\n")
        gui.action_label.config(text="行动：已弃牌，等待开局")
        Layer_Bg = LAYER_BG_FOLD
        #BoxText(f"日志：已弃牌，等待中...")
        BoxText(".")
        return False

    # 步骤 3：检查YOLO手牌和公牌是否变化，并提前执行AI识别
    yolo_result = detector.run_yolo_detection(screenshot, width, height)
    current_yolo = {'yo_hand': detector.normalize_cards(yolo_result['cards']['yo_hand']), 'yo_public': detector.normalize_cards(yolo_result['cards']['yo_public'])}

    # 比较当前YOLO结果与上一次是否相同（忽略顺序）| 检查手牌和公牌数量条件
    hand_changed = set(current_yolo['yo_hand']) != set(state.last_yolo_triggered['yo_hand'])
    public_changed = set(current_yolo['yo_public']) != set(state.last_yolo_triggered['yo_public'])
    hand_valid = hand_changed and current_yolo['yo_hand'] and len(current_yolo['yo_hand']) > 0
    public_valid = public_changed and len(current_yolo['yo_public']) > 1

    if len(current_yolo['yo_hand']) > 0 and (hand_valid or public_valid):
        BoxText(".")
        ai_result = detector.run_ai_analysis(screenshot)
        state.ai_hand = ai_result.get('hand', []) if ai_result.get('hand') else []
        state.ai_public = ai_result.get('public', []) if ai_result.get('public') else []
        state.last_yolo_triggered = current_yolo.copy()
        BoxText("b")

        # 检查是否为新牌局
        if state.ai_hand is not None and state.last_ai_triggered is not None and set(state.ai_hand) != set(state.last_ai_triggered):
            state.reset(exclude="|ai_hand|last_yolo_triggered|")
            state.last_ai_triggered = state.ai_hand.copy()

            if SHOW_LEFT_WINDOW:
                gui.decision_label.config(text="\n新局开始...\n")
            gui.action_label.config(text="行动：新局开始")
            BoxText("xj")
            if state.ai_hand is not None:
                state.last_ai_triggered = state.ai_hand.copy()
            else:
                state.last_ai_triggered = []
        elif state.ai_hand and not state.last_ai_triggered:  # 首次识别手牌
            state.reset(exclude="|ai_hand|last_yolo_triggered|")
            state.last_ai_triggered = state.ai_hand.copy()

    # 步骤 4：检查时钟标志是否 >= 2
    yolo_clock = yolo_result.get('clock', 0)
    if yolo_clock < 2:
        state.no_clock_count += 1
        if SHOW_LEFT_WINDOW:
            gui.decision_label.config(text="\n等待行动.\n")
        gui.action_label.config(text="行动：等待到我")
        BoxText("dw")
        return False
    state.no_clock_count = 0

    # 步骤 5：异步采集剩余数据
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_ocr = executor.submit(detector.run_ocr_tasks, screenshot, width, height)

        try:
            ocr_result = future_ocr.result()
        except Exception as e:
            BoxText(f"日志：OCR采集异常 | {str(e)}")
            ocr_result = {'chip': 0, 'bett': 0, 'hold': 0, 'blind': 0}

    # 更新状态
    state.mon_chip = ocr_result.get('chip', 0)
    state.mon_bett = ocr_result.get('bett', 0)
    state.mon_blind = ocr_result.get('blind', state.last_mon_blind)
    now_hold = ocr_result.get('hold', 0)
    
    # 安全的类型转换，防止JSON字符串或其他异常值
    try:
        if now_hold is not None and now_hold != '':
            # 尝试将可能的JSON字符串转换为数字
            hold_value = float(now_hold)
            if hold_value > 0:
                state.mon_hold = hold_value
    except (ValueError, TypeError) as e:
        BoxText(f"警告：hold值转换失败 '{now_hold}': {e}")
    state.player_count = yolo_result.get('player', 0)
    state.my_position = yolo_result.get('my_position', "未知位")
    state.after = yolo_result.get('after', 0)
    current_circle = get_current_circle(state.ai_public)
    current_history = yolo_result['history'].replace("_bett_", str(state.mon_bett)).replace("_circle_", current_circle)

    # 步骤 6：检查手牌是否为空
    if not state.ai_hand:
        if SHOW_LEFT_WINDOW:
            gui.decision_label.config(text="\n弃牌/等待中..\n")
        gui.action_label.config(text="行动：等待进入游戏")
        BoxText("日志：手牌为空，等待发牌")
        return False

    # 步骤 7：检查基础信息是否变化手牌公牌数量
    decision_key = (tuple(state.ai_hand) if state.ai_hand else (), tuple(state.ai_public) if state.ai_public else (), state.mon_chip, state.mon_hold, state.mon_bett)
    last_decision_key = (
    tuple(state.last_ai_triggered) if state.last_ai_triggered else (), tuple(state.last_public) if hasattr(state, 'last_public') else (), state.last_mon_chip, state.last_mon_hold, state.last_mon_bett)
    ai_hand_count = len(state.ai_hand) if state.ai_hand else 0
    ai_public_count = len(state.ai_public) if state.ai_public else 0
    valid_public_counts = {0, 3, 4, 5}

    if decision_key == last_decision_key and state.last_action is not None:
        #BoxText("日志：基础信息未变化，使用上一次决策")
        BoxText(".")
    elif ai_hand_count != 2 or ai_public_count not in valid_public_counts:
        BoxText(f"错误：跳过 | 手牌数量={ai_hand_count}（应为2），公牌数量={ai_public_count}（应为3/4/5），跳过异常序列")
        state.reset()  # 重置状态
        if SHOW_LEFT_WINDOW:
            gui.decision_label.config(text="\n异常：牌数量错误，重置中...\n")
        gui.action_label.config(text="行动：异常检测，重置")
        Layer_Bg = LAYER_BG_DEFAULT
        state.last_update_time = time.time()
        return False  # 不执行后续决策
    else:
        # 步骤 8：将数据传递给 MA_JC 进行决策
        state.action_count += 1
        history_entry = f"第{state.action_count}次行动，{current_history}"
        state.history.append(history_entry)
        history_str = "\n".join(state.history)
        action, betting, win, out_text = MA_JC(state.ai_hand, state.ai_public, state.mon_chip, state.mon_hold, state.mon_bett, state.mon_blind, state.my_position, state.after, history_str,
                                               state.player_count)
        state.last_action = action
        state.last_betting = betting
        state.last_out_text = out_text
        state.last_ai_triggered = state.ai_hand.copy()
        state.last_mon_chip = state.mon_chip
        state.last_mon_bett = state.mon_bett
        state.last_mon_blind = state.mon_blind
        state.last_mon_hold = state.mon_hold
        state.last_public = state.ai_public.copy()

        # 更新 history_str 中的 还未行动
        my_action_str = action
        if action in ["raise", "call", "error"] and betting is not None:
            my_action_str = f"{action} {betting}"
        current_history = current_history.replace("正在行动", my_action_str)
        state.history[-1] = f"第{state.action_count}次行动：{current_history}"  # 更新最新历史记录
        history_str = "\n".join(state.history)
        #
        # BoxText(f"日志：采集数据汇总 => 手牌={state.ai_hand} | 公牌={state.ai_public} | 锅底筹码={state.mon_chip} | "
        #         f"行动筹码={state.mon_bett} | 我的筹码={state.mon_hold} | 我的位置={state.my_position} | "
        #         f"在我后面行动人数={state.after} | 对手数量={state.player_count} | 历史记录={history_str}")
        BoxText(".")

        # 更新 UI
        if SHOW_LEFT_WINDOW:
            gui.decision_label.config(text=f"\n{out_text}\n")
        if action in ["raise", "call"]:
            Layer_Bg = LAYER_BG_RAISE_CALL
            gui.action_label.config(text=f"行动：{action} | 下注：{betting}")
            # await Play_Audio(f"{action}，{betting}")
        elif action == "check":
            Layer_Bg = LAYER_BG_DEFAULT
            gui.action_label.config(text=f"行动：{action}")
            # await Play_Audio(f"{action}")
        elif action == "fold":
            Layer_Bg = LAYER_BG_FOLD
            gui.action_label.config(text=f"行动：{action}")
            # await Play_Audio(f"{action}")
        else:
            gui.action_label.config(text=f"行动：{action}")
            # await Play_Audio(f"{action}")

    return True

def get_current_circle(ai_public):
    pc_count = len(ai_public) if isinstance(ai_public, list) else 0
    if pc_count == 0:
        return "底牌圈"
    elif pc_count == 3:
        return "翻牌圈"
    elif pc_count == 4:
        return "转牌圈"
    elif pc_count == 5:
        return "河牌圈"
    else:
        return "未知"

def update_gui_from_queue(gui, detector):
    global Layer_Bg
    try:
        current_time = time.time()
        if current_time - state.last_update_time > TIMEOUT_THRESHOLD:
            BoxText(f"警告：程序超过 {TIMEOUT_THRESHOLD} 秒未更新，正在重置状态")
            state.reset()
            if SHOW_LEFT_WINDOW:
                gui.decision_label.config(text="\n程序超时，重置中...\n")
            gui.action_label.config(text="行动：程序重置")
            Layer_Bg = LAYER_BG_DEFAULT
            state.last_update_time = current_time

    except Exception as e:
        BoxText(f"错误：GUI更新异常 | {Error_Detail(e)}")
        BoxText(f"调试：GUI异常重置前状态 | ai_hand={state.ai_hand}, last_ai_triggered={state.last_ai_triggered}")
        state.reset()
        if SHOW_LEFT_WINDOW:
            gui.decision_label.config(text="\n异常重置中...\n")
        gui.action_label.config(text="行动：异常重置")
        Layeralking_Bg = LAYER_BG_DEFAULT
        state.last_update_time = time.time()

    gui.update_layout()
    gui.root.after(WIN_AFTER, lambda: update_gui_from_queue(gui, detector))

def find_poker_window():
    for _ in range(100):
        hwnd = Win_Handle("Laixi.exe")
        if hwnd:
            return hwnd
        time.sleep(WIN_SLEEP)
    BoxText("未找到Laixi.exe窗口")
    return None

def main():
    hwnd = find_poker_window()
    if not hwnd:
        exit(1)

    global gui
    root = tk.Tk()
    gui = PokerGUI(root, hwnd)
    detector = PokerDetector(hwnd)

    # 注册窗口关闭事件处理
    def on_closing():
        detector.stop()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    update_gui_from_queue(gui, detector)
    
    try:
        root.mainloop()
    finally:
        # 确保无论如何都会清理资源
        detector.stop()

if __name__ == "__main__":
    try:
        if not Check_Auth():
            input("按任意键退出...")
            exit(1)
        else:
            amd, dmd, xmd, aipa, iden, rule, prefilter_enabled, correction_enabled = Read_Config(type="mode")
            print(f"***** 模型配置 *****")
            print(f"AI引擎：{aipa} (d=DeepSeek, a=阿里, x=xAI)")
            print(f"预过滤：{'启用' if prefilter_enabled else '禁用'} (阈值≤{rule}%)")
            print(f"纠错层：{'启用' if correction_enabled else '禁用'} (完全信任AI)")
            print(f"提示词：已加载优化版（结构化思维链）")
            print(f"***** end *****")
            main()
    except KeyboardInterrupt:
        print("\n")
        print("=" * 60)
        print("  程序已被用户手动终止 (Ctrl+C)")
        print("  感谢使用德州扑克AI助手！")
        print("=" * 60)
        sys.exit(0)
    except Exception as e:
        print("\n")
        print("=" * 60)
        print(f"  程序异常退出: {str(e)}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
