"""
pdocr.py - PaddleOCR-VL 本地推理模块
使用百度飞桨的 PaddleOCR-VL 模型进行视觉识别
作者：AI重构
日期：2024
"""

from core.comm import BoxText
import json
import time
import os


class PaddleOCRVL_Inference:
    """PaddleOCR-VL 本地推理器"""
    
    def __init__(self, vl_rec_backend='pytorch', vl_rec_server_url=None):
        """
        初始化 PaddleOCR-VL 模型
        
        Args:
            vl_rec_backend: 推理后端 ('pytorch' 或 'vllm-server')
            vl_rec_server_url: vLLM 服务器地址（如果使用 vllm-server）
        """
        self.pipeline = None
        self.is_loaded = False
        self.vl_rec_backend = vl_rec_backend
        self.vl_rec_server_url = vl_rec_server_url
        
        # 延迟加载（只在第一次使用时加载）
        # self._load_model()
    
    def _load_model(self):
        """加载模型（延迟加载）"""
        if self.is_loaded:
            return True
        
        try:
            BoxText("消息：开始加载 PaddleOCR-VL 模型...")
            start_time = time.time()
            
            # 检查依赖
            try:
                from paddleocr import PaddleOCRVL
                BoxText("✅ paddleocr 已安装")
            except ImportError:
                BoxText("错误：paddleocr 未安装")
                BoxText("提示：安装命令")
                BoxText("      pip install paddlepaddle-gpu==3.2.0")
                BoxText("      pip install -U paddleocr[doc-parser]")
                return False
            
            # 加载 pipeline
            BoxText(f"消息：初始化 PaddleOCR-VL pipeline（后端：{self.vl_rec_backend}）...")
            
            if self.vl_rec_backend == 'vllm-server' and self.vl_rec_server_url:
                # 使用 vLLM 服务器（更快）
                BoxText(f"消息：使用 vLLM 服务器 - {self.vl_rec_server_url}")
                self.pipeline = PaddleOCRVL(
                    vl_rec_backend=self.vl_rec_backend,
                    vl_rec_server_url=self.vl_rec_server_url
                )
            else:
                # 使用本地 PyTorch（默认）
                BoxText("消息：使用本地 PyTorch 推理")
                self.pipeline = PaddleOCRVL()
            
            load_time = time.time() - start_time
            BoxText(f"成功：PaddleOCR-VL 模型加载完成 - 耗时{load_time:.2f}秒")
            self.is_loaded = True
            return True
            
        except Exception as e:
            BoxText(f"错误：PaddleOCR-VL 模型加载失败 - {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def infer_poker(self, image_path):
        """
        德州扑克专用推理
        
        Args:
            image_path: 输入图片路径
            
        Returns:
            dict: 识别结果的JSON对象
        """
        # 延迟加载模型
        if not self.is_loaded:
            if not self._load_model():
                return None
        
        try:
            start_time = time.time()
            BoxText(f"消息：开始 PaddleOCR-VL 本地推理 - {image_path}")
            
            # 调用模型推理
            # PaddleOCR-VL 会自动识别文档中的所有内容
            output = self.pipeline.predict(image_path)
            
            infer_time = time.time() - start_time
            BoxText(f"成功：PaddleOCR-VL 推理完成 - 耗时{infer_time:.2f}秒")
            
            # 解析结果
            result = self._parse_result(output, image_path)
            return result
            
        except Exception as e:
            BoxText(f"错误：PaddleOCR-VL 推理失败 - {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _parse_result(self, output, image_path):
        """
        解析 PaddleOCR-VL 的输出结果，提取德州扑克相关信息
        
        Args:
            output: PaddleOCR-VL 的原始输出
            image_path: 图片路径
            
        Returns:
            dict: 解析后的JSON对象
        """
        try:
            BoxText("消息：解析 PaddleOCR-VL 输出...")
            
            # PaddleOCR-VL 返回的是文档解析结果
            # 我们需要从中提取德州扑克相关信息
            
            # 保存原始结果到临时文件用于调试
            temp_output_dir = "temp/paddleocr_output"
            os.makedirs(temp_output_dir, exist_ok=True)
            
            # 提取文本内容
            all_text = []
            for res in output:
                # res 包含识别的文本和结构
                # 尝试提取文本内容
                if hasattr(res, 'to_dict'):
                    res_dict = res.to_dict()
                    all_text.append(str(res_dict))
                else:
                    all_text.append(str(res))
            
            combined_text = '\n'.join(all_text)
            
            # 保存到文件用于调试
            debug_file = os.path.join(temp_output_dir, "raw_output.txt")
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(combined_text)
            
            BoxText(f"调试：PaddleOCR-VL 原始输出已保存到 {debug_file}")
            
            # 尝试从文本中提取德州扑克信息
            result = self._extract_poker_info(combined_text)
            
            BoxText(f"成功：识别结果 - 手牌{result.get('hand_cards')}, 公牌{result.get('public_cards')}, 底池{result.get('pot_chips')}")
            return result
            
        except Exception as e:
            BoxText(f"错误：解析 PaddleOCR-VL 结果失败 - {str(e)}")
            import traceback
            traceback.print_exc()
            return self._get_default_result()
    
    def _extract_poker_info(self, text):
        """
        从识别的文本中提取德州扑克信息
        
        这是一个简化版本，实际使用时可能需要更复杂的解析逻辑
        或者使用 LLM 来辅助解析
        
        Args:
            text: 识别的文本内容
            
        Returns:
            dict: 提取的扑克信息
        """
        import re
        
        result = self._get_default_result()
        
        # 这里需要实现具体的文本解析逻辑
        # 由于 PaddleOCR-VL 主要是做 OCR，返回的是文本
        # 我们需要从文本中提取结构化信息
        
        # 示例：尝试查找扑克牌符号
        # 实际使用时需要根据 PaddleOCR-VL 的实际输出格式来调整
        
        BoxText("警告：PaddleOCR-VL 目前返回通用 OCR 结果")
        BoxText("提示：建议使用阿里API或DeepSeek-OCR进行结构化识别")
        BoxText("      PaddleOCR-VL 更适合文档解析，不太适合游戏界面识别")
        
        return result
    
    def _get_default_result(self):
        """获取默认结果"""
        return {
            'hand_cards': [],
            'public_cards': [],
            'pot_chips': 1.0,
            'my_chips': 10.0,
            'bet_amount': 0.0,
            'blind': 0.1,
            'my_position': 'unknown',
            'game_stage': 'preflop'
        }
    
    def unload_model(self):
        """卸载模型（释放内存）"""
        if self.is_loaded:
            BoxText("消息：正在卸载 PaddleOCR-VL 模型...")
            del self.pipeline
            self.is_loaded = False
            BoxText("成功：PaddleOCR-VL 模型已卸载")


# 全局单例
_paddle_ocr_instance = None

def get_paddle_ocr_instance(vl_rec_backend='pytorch', vl_rec_server_url=None):
    """
    获取 PaddleOCR-VL 单例
    
    Args:
        vl_rec_backend: 推理后端
        vl_rec_server_url: vLLM 服务器地址
        
    Returns:
        PaddleOCRVL_Inference: PaddleOCR-VL实例
    """
    global _paddle_ocr_instance
    if _paddle_ocr_instance is None:
        _paddle_ocr_instance = PaddleOCRVL_Inference(
            vl_rec_backend=vl_rec_backend,
            vl_rec_server_url=vl_rec_server_url
        )
    return _paddle_ocr_instance


def infer_poker_paddle(image_path, vl_rec_backend='pytorch', vl_rec_server_url=None):
    """
    全局函数 - 德州扑克 PaddleOCR-VL 推理（对外接口）
    
    Args:
        image_path: 输入图片路径
        vl_rec_backend: 推理后端
        vl_rec_server_url: vLLM 服务器地址
        
    Returns:
        dict: 识别结果的JSON对象
    """
    paddle_ocr = get_paddle_ocr_instance(
        vl_rec_backend=vl_rec_backend,
        vl_rec_server_url=vl_rec_server_url
    )
    return paddle_ocr.infer_poker(image_path)


# 测试代码
if __name__ == "__main__":
    # 测试本地推理
    test_image = "temp/real_img.png"
    if os.path.exists(test_image):
        result = infer_poker_paddle(test_image)
        if result:
            print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"测试图片不存在: {test_image}")


