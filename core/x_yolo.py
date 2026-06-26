"""
x_yolo.py - YOLO检测模块（重构版）
检测牌面变化和决策时机
从原 yolo.py 平移核心功能
作者：AI重构
日期：2024
"""

from ultralytics import YOLO
import cv2
from core.comm import *
from collections import Counter
import torch
import warnings
import logging
import os
import sys

# 禁用警告和日志
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

# 环境设置
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Windows系统错误抑制
if sys.platform == 'win32':
    import ctypes
    kernel32 = ctypes.windll.kernel32
    SEM_FAILCRITICALERRORS = 0x0001
    SEM_NOGPFAULTERRORBOX = 0x0002
    SEM_NOOPENFILEERRORBOX = 0x8000
    kernel32.SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX)


class YoloDetector:
    """YOLO检测器 - 检测牌面变化和决策时机"""
    
    def __init__(self):
        """初始化YOLO模型"""
        # 加载模型
        self.Button_MODEL = YOLO(r"model\button.pt")
        if torch.cuda.is_available():
            self.Button_MODEL.to('cuda')
        else:
            self.Button_MODEL.to('cpu')
        
        BoxText("成功：YOLO模型加载完成")
    
    def detect_decision_moment(self, image_path):
        """
        检测是否需要做决策（核心函数）
        
        Args:
            image_path: 游戏窗口截图路径
            
        Returns:
            dict: {
                "need_decision": bool,  # 是否需要决策
                "clock_count": int,  # 时钟标志数量（>=2表示轮到我）
                "has_fold_button": bool,  # 是否显示弃牌按钮
                "player_count": int,  # 玩家数量
                "reason": str  # 判断原因
            }
        """
        try:
            # 检查图片是否存在
            if not os.path.exists(image_path):
                return {
                    "need_decision": False,
                    "clock_count": 0,
                    "has_fold_button": False,
                    "player_count": 0,
                    "reason": "图片不存在"
                }
            
            # 1. 首先检查是否已弃牌（最优先判断）
            has_fold = self._check_fold_status(image_path)
            if has_fold:
                return {
                    "need_decision": False,
                    "clock_count": 0,
                    "has_fold_button": False,
                    "player_count": 0,
                    "reason": "已弃牌，等待下一局"
                }
            
            # 2. 检测时钟标志（判断是否轮到我）
            clock_count, player_count = self._detect_clock_and_players(image_path)
            
            # 3. 判断是否需要决策
            need_decision = clock_count >= 2
            
            reason = ""
            if need_decision:
                reason = f"轮到我行动（检测到{clock_count}个时钟标志）"
            elif clock_count > 0:
                reason = f"等待轮到我（当前{clock_count}个时钟标志 < 2）"
            else:
                reason = "等待其他玩家行动"
            
            return {
                "need_decision": need_decision,
                "clock_count": clock_count,
                "has_fold_button": False,
                "player_count": player_count,
                "reason": reason
            }
            
        except Exception as e:
            BoxText(f"错误：决策时机检测失败 - {str(e)}")
            return {
                "need_decision": False,
                "clock_count": 0,
                "has_fold_button": False,
                "player_count": 0,
                "reason": f"检测异常: {str(e)}"
            }
    
    def _check_fold_status(self, image_path):
        """
        检查是否已弃牌
        
        Args:
            image_path: 游戏窗口截图路径
            
        Returns:
            bool: True表示已弃牌
        """
        try:
            # 读取图像并裁剪弃牌区域（手牌区域附近）
            image = cv2.imread(image_path)
            if image is None:
                return False
            
            height, width = image.shape[:2]
            # 裁剪手牌区域（屏幕下方中间）
            fold_region = image[
                int(height * 0.7):int(height * 0.852),
                int(width * 0.35):int(width * 0.65)
            ]
            
            # 保存临时图片
            fold_img_path = 'temp/fold_check.jpg'
            os.makedirs('temp', exist_ok=True)
            cv2.imwrite(fold_img_path, fold_region)
            
            # 使用YOLO检测弃牌标志
            confidence_min = 0.1
            self.Button_MODEL.conf = confidence_min
            
            orig_height, orig_width = fold_region.shape[:2]
            max_dim = max(orig_height, orig_width)
            imgsz = max(320, min(1280, (max_dim + 31) // 32 * 32))
            
            results = self.Button_MODEL.predict(
                source=fold_img_path,
                imgsz=imgsz,
                save=False,
                verbose=False
            )
            
            # 统计弃牌相关标志
            fold_labels = []
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    class_index = int(box.cls.item())
                    confidence = box.conf.item()
                    label = result.names.get(class_index, None).lower()
                    if confidence >= confidence_min:
                        fold_labels.append(label)
            
            # 判断是否有弃牌标志
            label_counts = Counter(fold_labels)
            qt_count = label_counts.get("s.qp", 0)  # 弃牌
            ddz_count = label_counts.get("s.ddz", 0)  # 等待中
            zl_count = label_counts.get("s.zl", 0)  # 站立
            
            fold_count = max(qt_count, ddz_count, zl_count)
            return fold_count > 0
            
        except Exception as e:
            BoxText(f"警告：弃牌状态检测异常 - {str(e)}")
            return False
    
    def _detect_clock_and_players(self, image_path):
        """
        检测时钟标志和玩家数量
        
        Args:
            image_path: 游戏窗口截图路径
            
        Returns:
            tuple: (clock_count, player_count)
                - clock_count: 时钟标志数量（>=2表示轮到我）
                - player_count: 玩家数量
        """
        try:
            confidence_min = 0.1
            self.Button_MODEL.conf = confidence_min
            
            original_image = cv2.imread(image_path)
            if original_image is None:
                return -1, -1
            
            orig_height, orig_width = original_image.shape[:2]
            max_dim = max(orig_height, orig_width)
            imgsz = max(320, min(1280, (max_dim + 31) // 32 * 32))
            
            results = self.Button_MODEL.predict(
                source=image_path,
                imgsz=imgsz,
                save=False,
                verbose=False
            )
            
            # 收集所有检测到的标签
            all_labels = []
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    class_index = int(box.cls.item())
                    confidence = box.conf.item()
                    label = result.names.get(class_index, None).lower()
                    if confidence >= confidence_min:
                        all_labels.append(label)
            
            if not all_labels:
                return -6, -6
            
            # 统计标签
            label_counts = Counter(all_labels)
            
            # 时钟标志（表示可以操作的按钮）
            clock_keys = ["b.nz", "b.33", "b.50", "b.75", "b.100", "b.120", "b.zyjz", "b.qp", "b.allyin"]
            clock_count = sum(label_counts.get(key, 0) for key in clock_keys)
            
            # 玩家数量（座位标志）- 使用更准确的统计方法
            player_count = self._count_players_accurate(label_counts)
            
            return clock_count, player_count
            
        except Exception as e:
            BoxText(f"错误：时钟检测失败 - {str(e)}")
            return -1, -1
    
    def _count_players_accurate(self, label_counts):
        """
        准确统计对手数量（只使用s.zw标签）
        
        s.zw 标签含义：对手数量（不包括自己，只计算未弃牌的对手）
        
        Args:
            label_counts: 标签计数器
            
        Returns:
            int: 对手数量（不包括自己）
        """
        # ======== 调试日志：显示所有检测到的标签 ========
        BoxText("=" * 60)
        BoxText("【YOLO标签检测详情】")
        BoxText("-" * 60)
        
        # 输出所有标签（按数量降序）
        if label_counts:
            BoxText("检测到的所有标签：")
            sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
            for label, count in sorted_labels:
                BoxText(f"  {label}: {count}个")
        else:
            BoxText("⚠️ 未检测到任何标签！")
        
        BoxText("-" * 60)
        
        # 只使用s.zw标签（对手数量，不包括自己）
        opponent_count = label_counts.get("s.zw", 0)
        
        # 如果检测失败，返回最小值1（至少1个对手）
        if opponent_count == 0:
            BoxText(f"⚠️  警告：未检测到s.zw标签，使用默认值1")
            opponent_count = 1
        
        # 限制在合理范围内（1-8个对手）
        opponent_count = max(1, min(opponent_count, 8))
        
        BoxText(f"✅ 对手数量 = {opponent_count}")
        BoxText("=" * 60)
        
        return opponent_count
    
    def detect_players_only(self, image_path):
        """
        专门用于检测玩家数量（用于修正VLM识别结果）
        
        Args:
            image_path: 游戏窗口截图路径
            
        Returns:
            int: 玩家数量，-1表示检测失败
        """
        try:
            _, player_count = self._detect_clock_and_players(image_path)
            return player_count
        except Exception as e:
            BoxText(f"错误：玩家数量检测失败 - {str(e)}")
            return -1


# 全局单例
_yolo_detector = None

def detect_decision_moment(image_path):
    """
    全局函数 - 检测是否需要做决策
    
    Args:
        image_path: 游戏窗口截图路径
        
    Returns:
        dict: 检测结果
    """
    global _yolo_detector
    if _yolo_detector is None:
        _yolo_detector = YoloDetector()
    return _yolo_detector.detect_decision_moment(image_path)

def detect_player_count(image_path):
    """
    全局函数 - 仅检测玩家数量（快速、准确）
    
    Args:
        image_path: 游戏窗口截图路径
        
    Returns:
        int: 玩家数量（包括自己），-1表示检测失败
    """
    global _yolo_detector
    if _yolo_detector is None:
        _yolo_detector = YoloDetector()
    return _yolo_detector.detect_players_only(image_path)


# 测试代码
if __name__ == "__main__":
    test_image = "temp/real_img.png"
    if os.path.exists(test_image):
        result = detect_decision_moment(test_image)
        print(f"检测结果: {result}")
    else:
        print(f"测试图片不存在: {test_image}")



