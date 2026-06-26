"""
x_vision.py - 视觉信息提取模块
支持三种模式：
1. 阿里云API模式：使用阿里云 qwen3-vl-flash API（推荐）
2. DeepSeek-OCR模式：使用本地部署的 DeepSeek-OCR
3. PaddleOCR-VL模式：使用百度飞桨 PaddleOCR-VL
作者：AI重构
日期：2024
"""

from openai import OpenAI
from core.comm import *
from core.config import *
import json
import time
import os
import cv2
import numpy as np

# 导入本地推理模块（延迟导入，避免不需要时加载）
_dsocr_module = None
_pdocr_module = None

def _get_dsocr():
    """延迟导入 DeepSeek-OCR 模块"""
    global _dsocr_module
    if _dsocr_module is None:
        from core.dsocr import infer_poker_local
        _dsocr_module = infer_poker_local
    return _dsocr_module

def _get_pdocr():
    """延迟导入 PaddleOCR-VL 模块"""
    global _pdocr_module
    if _pdocr_module is None:
        from core.pdocr import infer_poker_paddle
        _pdocr_module = infer_poker_paddle
    return _pdocr_module


class VisionAnalyzer:
    """
    视觉分析器 - 支持三种识别模式
    
    1. 阿里云API模式（al）：使用阿里云 qwen3-vl-flash API（快速、稳定、推荐）
    2. DeepSeek-OCR模式（dp）：使用本地部署的 DeepSeek-OCR（私密、无成本）
    3. PaddleOCR-VL模式（pd）：使用百度飞桨 PaddleOCR-VL（文档解析专用）
    """
    
    def __init__(self, vision_mode='al', local_model_name='deepseek-ai/DeepSeek-OCR', gpu_id='0'):
        """
        初始化视觉分析器
        
        Args:
            vision_mode: 识别模式 ('al'=阿里API, 'dp'=DeepSeek-OCR, 'pd'=PaddleOCR-VL)
            local_model_name: 本地模型名称或路径（仅dp/pd模式使用）
            gpu_id: GPU设备ID（仅dp/pd模式使用）
        """
        self.vision_mode = vision_mode.lower()
        self.local_model_name = local_model_name
        self.gpu_id = gpu_id
        
        # 模式验证
        valid_modes = ['al', 'dp', 'pd']
        if self.vision_mode not in valid_modes:
            BoxText(f"警告：无效的视觉模式 '{self.vision_mode}'，使用默认模式 'al'")
            self.vision_mode = 'al'
        
        # 根据模式初始化
        if self.vision_mode == 'al':
            BoxText("消息：视觉分析器 - 阿里云API模式（qwen3-vl-flash）")
            # 远程模式初始化阿里云客户端（密钥从 core/config.py 读取）
            self.Ali_Client = OpenAI(
                api_key=ALI_API_KEY,
                base_url=ALI_BASE_URL,
                timeout=30  # 增加超时时间到30秒
            )
            self.model = ALI_VISION_FLASH_MODEL  # 使用高精度视觉模型
        elif self.vision_mode == 'dp':
            BoxText("消息：视觉分析器 - DeepSeek-OCR 本地推理模式")
            # DeepSeek-OCR模式不需要初始化API客户端
            self.Ali_Client = None
            self.model = None
        elif self.vision_mode == 'pd':
            BoxText("消息：视觉分析器 - PaddleOCR-VL 本地推理模式")
            # PaddleOCR-VL模式不需要初始化API客户端
            self.Ali_Client = None
            self.model = None
        
    def _preprocess_image(self, image_path, output_path="temp/preprocessed.jpg"):
        """
        图片预处理优化 - 提升API响应速度
        
        优化措施：
        1. 降低分辨率到70%（VLM对分辨率不敏感）
        2. 裁剪顶部和底部无关区域（减少20%数据量）
        3. JPEG压缩到75质量（平衡质量和大小）
        
        Args:
            image_path: 原始图片路径
            output_path: 输出图片路径
            
        Returns:
            str: 预处理后的图片路径
        """
        try:
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                BoxText(f"警告：无法读取图片 {image_path}，使用原图")
                return image_path
            
            h, w = img.shape[:2]
            
            # 1. 裁剪无关区域（保留中间80%的高度区域）
            crop_top = int(h * 0.05)  # 去掉顶部5%
            crop_bottom = int(h * 0.95)  # 去掉底部5%
            img_cropped = img[crop_top:crop_bottom, :]
            
            # 2. 缩放到70%
            new_h = int(img_cropped.shape[0] * 0.7)
            new_w = int(img_cropped.shape[1] * 0.7)
            img_resized = cv2.resize(img_cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # 3. 保存为JPEG，质量75
            cv2.imwrite(output_path, img_resized, [cv2.IMWRITE_JPEG_QUALITY, 75])
            
            # 计算压缩比例
            original_size = os.path.getsize(image_path) / 1024  # KB
            new_size = os.path.getsize(output_path) / 1024  # KB
            compression_ratio = (1 - new_size / original_size) * 100
            
            BoxText(f"调试：[图片优化] 原始 {original_size:.1f}KB → 优化后 {new_size:.1f}KB (减少{compression_ratio:.0f}%)")
            
            return output_path
            
        except Exception as e:
            BoxText(f"警告：图片预处理失败 - {str(e)}，使用原图")
            return image_path
        
    def RTVI(self, image_path):
        """
        Real-Time Vision Intelligence - 实时视觉智能分析
        
        支持三种模式：
        1. 阿里云API模式（al）：调用阿里云API
        2. DeepSeek-OCR模式（dp）：调用本地DeepSeek-OCR
        3. PaddleOCR-VL模式（pd）：调用百度飞桨PaddleOCR-VL
        
        Args:
            image_path: 游戏窗口截图路径
            
        Returns:
            dict: 包含所有决策信息的JSON对象
        """
        import time
        start_time = time.time()
        
        try:
            BoxText(f"消息：开始视觉分析 - {image_path}")
            
            # 根据模式选择推理方法
            if self.vision_mode == 'al':
                return self._rtvi_remote(image_path, start_time)
            elif self.vision_mode == 'dp':
                return self._rtvi_local(image_path, start_time)
            elif self.vision_mode == 'pd':
                return self._rtvi_paddle(image_path, start_time)
            else:
                BoxText(f"错误：未知的视觉模式 '{self.vision_mode}'，使用默认模式")
                return self._rtvi_remote(image_path, start_time)
                
        except Exception as e:
            total_time = time.time() - start_time
            import traceback
            BoxText(f"错误：视觉分析失败 - 总耗时{total_time:.2f}秒")
            BoxText(f"错误类型：{type(e).__name__}")
            BoxText(f"错误信息：{str(e)}")
            BoxText(f"错误堆栈：")
            traceback.print_exc()
            return self._get_default_result()
    
    def _rtvi_local(self, image_path, start_time):
        """
        本地推理模式 - 使用 DeepSeek-OCR
        
        Args:
            image_path: 游戏窗口截图路径
            start_time: 开始时间
            
        Returns:
            dict: 识别结果
        """
        try:
            # 阶段1：检查图片文件
            step_start = time.time()
            if not os.path.exists(image_path):
                BoxText(f"错误：图片文件不存在 - {image_path}")
                return self._get_default_result()
            BoxText(f"调试：[阶段1] 图片文件检查完成 - 耗时{time.time()-step_start:.2f}秒")
            
            # 阶段2：本地推理使用原图（不预处理）
            step_start = time.time()
            BoxText(f"调试：[阶段2] 本地推理模式，使用原图...")
            preprocessed_path = image_path  # 直接使用原图
            BoxText(f"调试：[阶段2] 跳过预处理 - 耗时{time.time()-step_start:.2f}秒")
            
            # 阶段3：调用本地推理
            step_start = time.time()
            BoxText(f"调试：[阶段3] 开始本地推理（DeepSeek-OCR）...")
            
            infer_func = _get_dsocr()
            result = infer_func(
                preprocessed_path,
                model_name=self.local_model_name,
                gpu_id=self.gpu_id
            )
            
            if result is None:
                BoxText(f"错误：本地推理返回空结果")
                return self._get_default_result()
            
            BoxText(f"调试：[阶段3] 本地推理完成 - 耗时{time.time()-step_start:.2f}秒")
            
            # 阶段4：数据验证和清洗
            step_start = time.time()
            BoxText(f"调试：[阶段4] 开始数据清洗和验证...")
            result = self._validate_and_clean(result)
            BoxText(f"调试：[阶段4] 数据清洗完成 - 耗时{time.time()-step_start:.2f}秒")
            
            # 总耗时
            total_time = time.time() - start_time
            BoxText(f"成功：视觉分析完成（本地模式）- 总耗时{total_time:.2f}秒")
            BoxText(f"成功：识别结果 - 手牌{result.get('hand_cards')}, 公牌{result.get('public_cards')}, 底池{result.get('pot_chips')}")
            
            return result
            
        except Exception as e:
            BoxText(f"错误：本地推理失败 - {str(e)}")
            import traceback
            traceback.print_exc()
            return self._get_default_result()
    
    def _rtvi_paddle(self, image_path, start_time):
        """
        PaddleOCR-VL 推理模式 - 使用百度飞桨 PaddleOCR-VL
        
        Args:
            image_path: 游戏窗口截图路径
            start_time: 开始时间
            
        Returns:
            dict: 识别结果
        """
        try:
            # 阶段1：检查图片文件
            step_start = time.time()
            if not os.path.exists(image_path):
                BoxText(f"错误：图片文件不存在 - {image_path}")
                return self._get_default_result()
            BoxText(f"调试：[阶段1] 图片文件检查完成 - 耗时{time.time()-step_start:.2f}秒")
            
            # 阶段2：调用 PaddleOCR-VL 推理
            step_start = time.time()
            BoxText(f"调试：[阶段2] 开始 PaddleOCR-VL 推理...")
            
            infer_func = _get_pdocr()
            result = infer_func(image_path)
            
            if result is None:
                BoxText(f"错误：PaddleOCR-VL 推理返回空结果")
                return self._get_default_result()
            
            BoxText(f"调试：[阶段2] PaddleOCR-VL 推理完成 - 耗时{time.time()-step_start:.2f}秒")
            
            # 阶段3：数据验证和清洗
            step_start = time.time()
            BoxText(f"调试：[阶段3] 开始数据清洗和验证...")
            result = self._validate_and_clean(result)
            BoxText(f"调试：[阶段3] 数据清洗完成 - 耗时{time.time()-step_start:.2f}秒")
            
            # 总耗时
            total_time = time.time() - start_time
            BoxText(f"成功：视觉分析完成（PaddleOCR-VL模式）- 总耗时{total_time:.2f}秒")
            BoxText(f"成功：识别结果 - 手牌{result.get('hand_cards')}, 公牌{result.get('public_cards')}, 底池{result.get('pot_chips')}")
            
            return result
            
        except Exception as e:
            BoxText(f"错误：PaddleOCR-VL 推理失败 - {str(e)}")
            import traceback
            traceback.print_exc()
            return self._get_default_result()
    
    def _rtvi_remote(self, image_path, start_time):
        """
        远程API模式 - 使用阿里云API
        
        Args:
            image_path: 游戏窗口截图路径
            start_time: 开始时间
            
        Returns:
            dict: 识别结果
        """
        try:
            BoxText(f"消息：使用远程API模式")
            
            # 阶段1：检查图片文件
            step_start = time.time()
            if not os.path.exists(image_path):
                BoxText(f"错误：图片文件不存在 - {image_path}")
                return self._get_default_result()
            BoxText(f"调试：[阶段1] 图片文件检查完成 - 耗时{time.time()-step_start:.2f}秒")
            
            # 构建简化的提示词 - 只识别核心决策信息（玩家信息由YOLO提供）
            system_prompt = """你是德州扑克游戏的专业视觉分析专家，专注于快速、准确地识别游戏核心信息。

【扑克牌表示规则】
- 点数：2-9用数字，10用T，J/Q/K/A用字母
- 花色：s=黑桃♠, h=红桃♥, c=梅花♣, d=方块♦
- 示例：Ks（黑桃K）、Th（红桃10）、9c（梅花9）

【关键区域位置 - 精确定位】
1. **我的手牌**：屏幕最底部中央，两张底牌
2. **公共牌**：屏幕中央偏上，横向排列3-5张牌
3. **底池金额**：★最重要★ 公牌正下方绿色长条区域的数字
4. **跟注金额**：★最容易出错的字段！★ 手牌右侧的大圆形按钮
   - 视觉寻找步骤：
     第一步：定位手牌（屏幕底部中央的两张牌）
     第二步：从手牌向右看，找到紧邻手牌的大圆形按钮
     第三步：仔细观察该按钮的颜色和内容
   - 按钮位置：紧贴手牌右侧，距离手牌最近的大圆形按钮
   - 识别按钮内容（仔细看！）：
     · 绿色按钮写着"让牌" → 返回0
     · 蓝色按钮显示数字"1" → 返回1（不要误判为0！）
     · 蓝色按钮显示数字"1.2" → 返回1.2
     · 没有这样的按钮 → 返回0
   - 警告：必须仔细识别蓝色按钮上的数字，不要错误地返回0！
5. **我的筹码**：手牌下方的数字（通常在黑色圆角矩形中）
6. **大盲注**：屏幕上方的盲注信息（格式如"盲注:0.2/0.5/1"）
7. **我的位置**：根据庄家D标记推断（Button/SB/BB/UTG/CO/HJ等）

【位置识别】
从庄家D开始顺时针：Button → SB → BB → UTG → CO → HJ"""

            user_prompt = """请快速提取德州扑克游戏截图中的核心决策信息（玩家数量已由系统提供，无需识别）。

【必须识别的核心信息】
1. **我的手牌**：屏幕底部中央两张底牌
   - 示例：["Ks", "9c"]

2. **公共牌**：屏幕中央横向排列的牌（0-5张）
   - 即使模糊也要尽力识别
   - 有公牌必须返回，不能返回空数组
   - 示例：["Ah", "Kd", "7s"]

3. **底池金额**：★最重要★ 公牌正下方绿色长条区域的数字
   - 示例：8.6

4. **我的筹码**：手牌下方或旁边的剩余筹码
   - 示例：42.5

5. **跟注金额**：★★★ 找到手牌右侧的圆形按钮，读取上面的数字！
   - 寻找位置：从手牌（底部中央两张牌）向右看，找到紧邻手牌的第一个大圆形按钮
   - 该按钮的特征：
     · 位置：紧贴手牌右侧
     · 大小：大圆形按钮，比其他按钮更大更显眼
     · 颜色：可能是蓝色或绿色
   - 识别该按钮上的内容：
     a) 如果按钮是绿色，上面写着"让牌" → 返回0
     b) 如果按钮是蓝色，上面显示一个数字（如"1"、"1.2"、"0.5"等） → 返回该数字
     c) 如果没有找到这样的按钮 → 返回0
   - 关键提醒：
     · 这个按钮通常位于手牌和"自由加注"按钮之间
     · 蓝色按钮上的数字就是需要跟注的金额，必须准确识别
     · 不要看手牌上方其他玩家位置的数字
   - 案例1：手牌右侧大圆形按钮是绿色"让牌" → 返回0
   - 案例2：手牌右侧大圆形按钮是蓝色显示"1" → 返回1（不要返回0！）
   - 案例3：手牌右侧大圆形按钮是蓝色显示"1.2" → 返回1.2

6. **大盲注**：屏幕上方的盲注信息，通常格式为"盲注:X/Y/Z"
   - 取最后一个数字作为大盲注
   - 示例：如果显示"盲注:0.2/0.5/1"，则大盲注为1

7. **我的位置**：根据庄家D标记推断
   - 示例："Button"、"SB"、"BB"、"UTG"、"CO"、"HJ"

8. **游戏阶段**：根据公牌数量判断
   - 0张=preflop, 3张=flop, 4张=turn, 5张=river

【输出格式】
严格按照JSON格式返回，不要有任何说明文字：

{
    "hand_cards": ["牌1", "牌2"],
    "public_cards": ["牌1", "牌2", "牌3"],
    "pot_chips": 数字,
    "my_chips": 数字,
    "bet_amount": 数字,
    "blind": 数字,
    "my_position": "位置名称",
    "game_stage": "preflop/flop/turn/river"
}

【识别要点 - 务必准确识别跟注金额】
1. ★★★ **底池金额**：公牌正下方绿色长条区域的数字
2. ★★★ **跟注金额 - 手牌右侧的大圆形按钮**
   - ⚠️ 寻找方法：从手牌向右看 → 找到紧邻的大圆形按钮 → 读取按钮上的内容
   - ⚠️ 按钮特征：
     · 位置：紧贴手牌右侧（不是远处的按钮）
     · 大小：大圆形按钮，很显眼
     · 内容：绿色"让牌"文字 或 蓝色数字（如"1"、"1.2"）
   - ⚠️ 识别规则：
     · 绿色"让牌"按钮 → bet_amount = 0
     · 蓝色按钮显示"1" → bet_amount = 1（不要错误返回0！）
     · 蓝色按钮显示"1.2" → bet_amount = 1.2
   - ⚠️ 常见错误1：手牌右侧明明有蓝色按钮显示"1"，却错误返回0
   - ⚠️ 常见错误2：看到远处的数字，忽略了紧邻手牌的按钮
   - ⚠️ 记住：必须仔细检查手牌右侧的大圆形按钮上显示的内容！
3. ★★ **公牌识别**：即使模糊也要尽力识别，有公牌必须返回
4. ★★ **大盲注提取**：从"盲注:X/Y/Z"中取最后一个数字Z
5. ★ **花色准确性**：仔细区分 s/h/c/d
6. ★ **纯数字**：金额不要包含$、￥等符号

只返回JSON，不要有任何解释。"""

            # 阶段2：图片预处理 + 转换为base64
            step_start = time.time()
            if not image_path.startswith("http"):
                BoxText(f"调试：[阶段2] 开始图片预处理...")
                # 预处理图片（缩放、裁剪、压缩）
                preprocessed_path = self._preprocess_image(image_path)
                # 转换为base64
                image_data = f"data:image/jpeg;base64,{Img_Base64(img=preprocessed_path)}"
                BoxText(f"调试：[阶段2] 预处理+base64完成 - 耗时{time.time()-step_start:.2f}秒")
            else:
                image_data = image_path
                BoxText(f"调试：[阶段2] 使用URL图片 - {image_path}")
            
            # 阶段3：调用阿里云多模态API（增加重试机制）
            BoxText(f"调试：[阶段3] 准备调用阿里云API - 模型{self.model}")
            api_start = time.time()
            max_retries = 2
            
            for retry in range(max_retries):
                try:
                    retry_start = time.time()
                    BoxText(f"调试：[阶段3] 第{retry+1}次API调用开始...")
                    
                    completion = self.Ali_Client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system", 
                                "content": [{"type": "text", "text": system_prompt}]
                            },
                            {
                                "role": "user", 
                                "content": [
                                    {"type": "image_url", "image_url": {"url": image_data}},
                                    {"type": "text", "text": user_prompt}
                                ]
                            }
                        ],
                        temperature=0,  # 使用确定性输出
                        response_format={'type': 'json_object'}
                    )
                    
                    BoxText(f"调试：[阶段3] API调用成功 - 耗时{time.time()-retry_start:.2f}秒")
                    break  # 成功则跳出重试循环
                    
                except Exception as e:
                    error_time = time.time() - retry_start
                    BoxText(f"警告：[阶段3] 第{retry+1}次API调用失败 - 耗时{error_time:.2f}秒")
                    BoxText(f"错误详情：{type(e).__name__} - {str(e)}")
                    
                    if retry < max_retries - 1:
                        BoxText(f"调试：[阶段3] 等待1秒后重试...")
                        time.sleep(1)
                        continue
                    else:
                        BoxText(f"错误：[阶段3] API调用失败，已重试{max_retries}次")
                        raise e  # 最后一次重试失败则抛出异常
            
            BoxText(f"调试：[阶段3] API调用总耗时{time.time()-api_start:.2f}秒")
            
            # 阶段4：解析返回结果
            step_start = time.time()
            BoxText(f"调试：[阶段4] 开始解析API返回结果...")
            
            result_text = completion.choices[0].message.content
            BoxText(f"调试：[阶段4] 原始返回长度：{len(result_text)}字符")
            
            result = json.loads(result_text)
            BoxText(f"调试：[阶段4] JSON解析成功 - 耗时{time.time()-step_start:.2f}秒")
            
            # 阶段5：数据清洗和验证
            step_start = time.time()
            BoxText(f"调试：[阶段5] 开始数据清洗和验证...")
            result = self._validate_and_clean(result)
            BoxText(f"调试：[阶段5] 数据清洗完成 - 耗时{time.time()-step_start:.2f}秒")
            
            # 总耗时
            total_time = time.time() - start_time
            BoxText(f"成功：视觉分析完成 - 总耗时{total_time:.2f}秒")
            BoxText(f"成功：识别结果 - 手牌{result.get('hand_cards')}, 公牌{result.get('public_cards')}, 底池{result.get('pot_chips')}")
            
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            import traceback
            BoxText(f"错误：视觉分析失败 - 总耗时{total_time:.2f}秒")
            BoxText(f"错误类型：{type(e).__name__}")
            BoxText(f"错误信息：{str(e)}")
            BoxText(f"错误堆栈：")
            traceback.print_exc()
            return self._get_default_result()
    
    def _validate_and_clean(self, result):
        """验证和清洗AI返回的数据"""
        try:
            # 确保必要字段存在（简化版，移除玩家相关字段）
            defaults = {
                "hand_cards": [],
                "public_cards": [],
                "pot_chips": 1.0,
                "my_chips": 10.0,
                "bet_amount": 0.0,
                "blind": 0.1,
                "my_position": "unknown",
                "game_stage": "preflop"
            }
            
            for key, default_value in defaults.items():
                if key not in result:
                    result[key] = default_value
            
            # 标准化牌面表示
            if result["hand_cards"]:
                result["hand_cards"] = [self._normalize_card(card) for card in result["hand_cards"]]
            if result["public_cards"]:
                result["public_cards"] = [self._normalize_card(card) for card in result["public_cards"]]
            
            # 确保数值类型正确
            result["pot_chips"] = float(result["pot_chips"]) if result["pot_chips"] else 1.0
            result["my_chips"] = float(result["my_chips"]) if result["my_chips"] else 10.0
            result["bet_amount"] = float(result["bet_amount"]) if result["bet_amount"] else 0.0
            result["blind"] = float(result["blind"]) if result["blind"] else 0.1
            
            return result
            
        except Exception as e:
            BoxText(f"警告：数据清洗失败 - {str(e)}")
            return self._get_default_result()
    
    def _normalize_card(self, card):
        """标准化牌面表示，统一为 点数大写+花色小写 格式（如：Ks, 6c, Tc, Ad）"""
        if not card or len(card) < 2:
            return card
        
        # 移除可能的符号
        card = str(card).replace('♠', 's').replace('♥', 'h').replace('♣', 'c').replace('♦', 'd')
        card = card.replace('spade', 's').replace('heart', 'h').replace('club', 'c').replace('diamond', 'd')
        
        # 提取点数和花色
        rank = card[:-1].upper()  # 点数转大写（10要特殊处理）
        suit = card[-1].lower()   # 花色转小写
        
        # 10转换为T
        if rank == '10':
            rank = 'T'
        
        # 验证格式
        valid_ranks = '23456789TJQKA'
        valid_suits = 'shdc'
        
        if rank in valid_ranks and suit in valid_suits:
            return f"{rank}{suit}"
        
        BoxText(f"警告：无效的牌面格式 '{card}'，已忽略")
        return card
    
    def _get_default_result(self):
        """返回默认结果（当分析失败时）"""
        return {
            "hand_cards": [],
            "public_cards": [],
            "pot_chips": 1.0,
            "my_chips": 10.0,
            "bet_amount": 0.0,
            "blind": 0.1,
            "my_position": "unknown",
            "game_stage": "preflop",
            "error": "视觉分析失败，使用默认值"
        }


# 全局单例和配置
_vision_analyzer = None
_vision_config = {
    'vision_mode': 'al',  # 默认使用阿里云API ('al', 'dp', 'pd')
    'local_model_name': 'deepseek-ai/DeepSeek-OCR',
    'gpu_id': '0'
}

def set_vision_config(vision_mode='al', local_model_name='deepseek-ai/DeepSeek-OCR', gpu_id='0'):
    """
    配置视觉分析器
    
    Args:
        vision_mode: 识别模式 ('al'=阿里API, 'dp'=DeepSeek-OCR, 'pd'=PaddleOCR-VL)
        local_model_name: 本地模型名称或路径（仅dp/pd模式使用）
        gpu_id: GPU设备ID（仅dp/pd模式使用）
    """
    global _vision_config, _vision_analyzer
    _vision_config = {
        'vision_mode': vision_mode,
        'local_model_name': local_model_name,
        'gpu_id': gpu_id
    }
    # 重置单例，让下次调用时使用新配置
    _vision_analyzer = None
    
    # 显示模式信息
    mode_map = {
        'al': '阿里云API模式（qwen3-vl-flash）',
        'dp': 'DeepSeek-OCR 本地推理模式',
        'pd': 'PaddleOCR-VL 本地推理模式'
    }
    mode_text = mode_map.get(vision_mode.lower(), f'未知模式({vision_mode})')
    BoxText(f"消息：视觉分析器配置已更新 - {mode_text}")

def RTVI(image_path):
    """
    全局函数 - Real-Time Vision Intelligence
    一次性提取所有决策所需信息
    
    支持三种模式（通过 set_vision_config 配置）：
    1. 阿里云API模式（默认）：使用阿里云 qwen3-vl-flash
    2. DeepSeek-OCR模式：使用本地 DeepSeek-OCR
    3. PaddleOCR-VL模式：使用百度飞桨 PaddleOCR-VL
    
    Args:
        image_path: 游戏窗口截图路径
        
    Returns:
        dict: 包含所有决策信息的字典
    """
    global _vision_analyzer
    if _vision_analyzer is None:
        _vision_analyzer = VisionAnalyzer(
            vision_mode=_vision_config['vision_mode'],
            local_model_name=_vision_config['local_model_name'],
            gpu_id=_vision_config['gpu_id']
        )
    return _vision_analyzer.RTVI(image_path)


# 测试代码
if __name__ == "__main__":
    # 测试视觉分析功能
    test_image = "temp/real_img.png"
    if os.path.exists(test_image):
        result = RTVI(test_image)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"测试图片不存在: {test_image}")

