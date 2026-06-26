"""
dsocr.py - DeepSeek-OCR 本地推理模块
使用本地部署的 DeepSeek-OCR 模型进行视觉识别
作者：AI重构
日期：2024
"""

from transformers import AutoModel, AutoTokenizer
import torch
import os
import time
from core.comm import BoxText
import json


class DeepSeekOCR:
    """DeepSeek-OCR 本地推理器"""
    
    def __init__(self, model_name='deepseek-ai/DeepSeek-OCR', gpu_id='0'):
        """
        初始化 DeepSeek-OCR 模型
        
        Args:
            model_name: 模型名称或本地路径
            gpu_id: GPU设备ID
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.gpu_id = gpu_id
        self.is_loaded = False
        
        # 延迟加载（只在第一次使用时加载模型）
        # self._load_model()
    
    def _load_model(self):
        """加载模型（延迟加载）"""
        if self.is_loaded:
            return
        
        try:
            BoxText("消息：开始加载 DeepSeek-OCR 本地模型...")
            start_time = time.time()
            
            # 设置GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_id
            
            # 检查CUDA是否可用
            if not torch.cuda.is_available():
                BoxText("错误：CUDA不可用，DeepSeek-OCR需要GPU支持")
                return False
            
            BoxText(f"消息：使用GPU: {torch.cuda.get_device_name(0)}")
            
            # 加载tokenizer
            BoxText("消息：加载tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            # 加载模型
            BoxText("消息：加载模型（这可能需要几分钟，首次运行会下载模型）...")
            
            # 尝试使用 Flash Attention，如果失败则降级到普通模式
            try:
                BoxText("消息：尝试使用 Flash Attention 2（更快）...")
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    _attn_implementation='flash_attention_2',
                    trust_remote_code=True,
                    use_safetensors=True
                )
                BoxText("成功：使用 Flash Attention 2 模式")
            except ImportError as e:
                BoxText("警告：Flash Attention 2 未安装，降级到普通模式")
                BoxText("提示：安装 Flash Attention 可提升推理速度")
                BoxText("      运行：pip install flash-attn --no-build-isolation")
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    use_safetensors=True
                )
                BoxText("成功：使用普通 Attention 模式（速度较慢但可用）")
            
            # 设置为评估模式并移到GPU
            self.model = self.model.eval().cuda().to(torch.bfloat16)
            
            # 修改生成配置，增加最大生成长度
            if hasattr(self.model, 'generation_config'):
                BoxText("消息：修改生成配置，增加最大生成长度...")
                self.model.generation_config.max_new_tokens = 1024  # 进一步增加
                self.model.generation_config.max_length = 4096
                # 禁用一些可能限制生成的选项
                if hasattr(self.model.generation_config, 'early_stopping'):
                    self.model.generation_config.early_stopping = False
                BoxText(f"调试：当前生成配置 - max_new_tokens={self.model.generation_config.max_new_tokens}")
            
            # 尝试猴子补丁：修改模型的默认生成参数
            try:
                original_generate = self.model.generate
                def patched_generate(*args, **kwargs):
                    # 强制设置生成参数
                    kwargs.setdefault('max_new_tokens', 1024)
                    kwargs.setdefault('min_new_tokens', 10)  # 至少生成10个token
                    return original_generate(*args, **kwargs)
                self.model.generate = patched_generate
                BoxText("消息：已应用生成参数补丁")
            except Exception as e:
                BoxText(f"警告：无法应用生成参数补丁 - {e}")
            
            load_time = time.time() - start_time
            BoxText(f"成功：DeepSeek-OCR 模型加载完成 - 耗时{load_time:.2f}秒")
            self.is_loaded = True
            return True
            
        except Exception as e:
            BoxText(f"错误：DeepSeek-OCR 模型加载失败 - {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _direct_generate(self, image_path, prompt):
        """
        直接使用模型的generate方法（备用方案）
        
        Args:
            image_path: 输入图片路径
            prompt: 提示词
            
        Returns:
            str: 生成的文本
        """
        try:
            BoxText("调试：尝试使用直接生成方法...")
            
            # 准备输入
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
            
            # 编码输入
            inputs = self.model.prepare_inputs(
                prompt=prompt,
                image=image,
                tokenizer=self.tokenizer
            )
            
            # 移动到GPU
            inputs = {k: v.to(self.model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,  # 增加最大生成长度
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    num_beams=1,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # 解码输出
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取prompt之后的部分
            if prompt in response:
                response = response.split(prompt)[-1].strip()
            
            BoxText(f"调试：直接生成方法返回 - {response}")
            return response
            
        except Exception as e:
            BoxText(f"警告：直接生成方法失败 - {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def infer_poker(self, image_path, output_path="temp/dsocr_output"):
        """
        德州扑克专用推理
        
        Args:
            image_path: 输入图片路径
            output_path: 输出路径
            
        Returns:
            dict: 识别结果的JSON对象
        """
        # 延迟加载模型
        if not self.is_loaded:
            if not self._load_model():
                return None
        
        try:
            start_time = time.time()
            BoxText(f"消息：开始 DeepSeek-OCR 本地推理 - {image_path}")
            
            # 构建德州扑克专用prompt
            # DeepSeek-OCR更适合简单的OCR任务，而不是复杂的结构化提取
            # 先尝试最简单的prompt
            prompt = """<image>
请详细描述图片中的所有文字、数字和扑克牌。"""
            
            # 调用模型推理
            # 确保输出目录存在
            os.makedirs(output_path, exist_ok=True)
            
            BoxText(f"调试：推理参数 - base_size=1024, image_size=640, crop_mode=True")
            BoxText(f"调试：prompt长度 - {len(prompt)} 字符")
            BoxText(f"调试：图片路径 - {image_path}")
            BoxText(f"调试：输出路径 - {output_path}")
            
            # 检查infer方法的签名
            import inspect
            if hasattr(self.model, 'infer'):
                sig = inspect.signature(self.model.infer)
                BoxText(f"调试：infer方法签名 - {sig}")
            
            # Gundam配置：base_size = 1024, image_size = 640, crop_mode = True
            # 尝试捕获标准输出
            import sys
            from io import StringIO
            
            # 保存原始stdout
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                # 只使用infer方法支持的参数
                res = self.model.infer(
                    self.tokenizer,
                    prompt=prompt,
                    image_file=image_path,
                    output_path=output_path,
                    base_size=1024,
                    image_size=640,
                    crop_mode=True,
                    test_compress=False,
                    save_results=True,
                    eval_mode=False  # 尝试eval_mode参数
                )
                
                # 获取捕获的输出
                captured_output = sys.stdout.getvalue()
            finally:
                # 恢复stdout
                sys.stdout = old_stdout
            
            # 打印捕获的输出
            if captured_output:
                BoxText(f"调试：模型标准输出:\n{captured_output}")
            
            infer_time = time.time() - start_time
            BoxText(f"成功：DeepSeek-OCR 推理完成 - 耗时{infer_time:.2f}秒")
            
            # 调试：打印返回值类型和内容
            BoxText(f"调试：模型返回值类型 - {type(res)}")
            BoxText(f"调试：模型返回值内容 - {res}")
            
            # 如果返回结果为None，尝试从输出文件中读取
            if res is None:
                BoxText("调试：返回值为None，尝试从输出目录读取结果...")
                
                # 列出输出目录中的所有文件
                if os.path.exists(output_path):
                    files = os.listdir(output_path)
                    BoxText(f"调试：输出目录 {output_path} 中的文件: {files}")
                    
                    # 尝试读取所有可能的结果文件
                    possible_files = [
                        "result.mmd",  # DeepSeek-OCR 的默认输出文件
                        "results.txt", "result.txt", "output.txt", "response.txt",
                        "results.json", "result.json", "output.json"
                    ]
                    
                    for filename in possible_files:
                        result_file = os.path.join(output_path, filename)
                        if os.path.exists(result_file):
                            with open(result_file, 'r', encoding='utf-8') as f:
                                res = f.read()
                            # 限制显示长度以避免日志过长
                            display_text = res if len(res) <= 1000 else f"{res[:500]}...(共{len(res)}字符)...{res[-500:]}"
                            BoxText(f"调试：从文件 {filename} 读取到的内容 - {display_text}")
                            break
                    
                    # 如果还是没找到，尝试读取目录中第一个txt、json或mmd文件
                    if res is None and files:
                        for filename in files:
                            if filename.endswith(('.txt', '.json', '.mmd')):
                                result_file = os.path.join(output_path, filename)
                                with open(result_file, 'r', encoding='utf-8') as f:
                                    res = f.read()
                                BoxText(f"调试：从文件 {filename} 读取到的内容（前500字符） - {res[:500]}")
                                break
                else:
                    BoxText(f"调试：输出目录不存在 - {output_path}")
            
            # 如果还是None，尝试直接生成方法
            if res is None:
                BoxText("警告：无法从infer方法获取结果，尝试使用直接生成方法...")
                res = self._direct_generate(image_path, prompt)
            
            # 解析结果
            result = self._parse_result(res)
            return result
            
        except Exception as e:
            BoxText(f"错误：DeepSeek-OCR 推理失败 - {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _parse_result(self, res):
        """
        解析模型返回的结果
        
        Args:
            res: 模型返回的原始结果
            
        Returns:
            dict: 解析后的JSON对象
        """
        try:
            # DeepSeek-OCR返回的可能是元组、列表、字符串或字典
            text_content = None
            
            # 处理None
            if res is None:
                BoxText(f"警告：模型返回结果为None")
                return None
            
            # 处理元组或列表（提取第一个元素作为文本）
            elif isinstance(res, (tuple, list)):
                if len(res) > 0:
                    text_content = res[0]
                    BoxText(f"调试：从元组/列表中提取文本 - {text_content}")
                else:
                    BoxText(f"警告：返回的元组/列表为空")
                    return None
            
            # 处理字典
            elif isinstance(res, dict):
                # 如果已经是字典，检查是否包含必要字段
                if 'hand_cards' in res:
                    result = res
                # 否则可能字典中有text字段
                elif 'text' in res:
                    text_content = res['text']
                else:
                    BoxText(f"警告：字典中没有找到预期的字段")
                    BoxText(f"字典内容：{res}")
                    return None
            
            # 处理字符串
            elif isinstance(res, str):
                text_content = res
            
            # 未知类型
            else:
                BoxText(f"警告：未知的返回结果类型: {type(res)}")
                BoxText(f"返回结果：{res}")
                return None
            
            # 如果提取到了文本内容，尝试解析JSON
            if text_content is not None:
                # 查找JSON部分
                start_idx = text_content.find('{')
                end_idx = text_content.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = text_content[start_idx:end_idx]
                    result = json.loads(json_str)
                else:
                    BoxText(f"警告：无法从返回文本中提取JSON")
                    BoxText(f"返回文本：{text_content}")
                    return None
            
            # 验证必要字段
            required_fields = ['hand_cards', 'public_cards', 'pot_chips', 'my_chips', 
                              'bet_amount', 'blind', 'my_position', 'game_stage']
            for field in required_fields:
                if field not in result:
                    result[field] = self._get_default_value(field)
            
            BoxText(f"成功：识别结果 - 手牌{result.get('hand_cards')}, 公牌{result.get('public_cards')}, 底池{result.get('pot_chips')}")
            return result
            
        except Exception as e:
            BoxText(f"错误：解析 DeepSeek-OCR 结果失败 - {str(e)}")
            return None
    
    def _get_default_value(self, field):
        """获取字段的默认值"""
        defaults = {
            'hand_cards': [],
            'public_cards': [],
            'pot_chips': 1.0,
            'my_chips': 10.0,
            'bet_amount': 0.0,
            'blind': 0.1,
            'my_position': 'unknown',
            'game_stage': 'preflop'
        }
        return defaults.get(field)
    
    def unload_model(self):
        """卸载模型（释放GPU内存）"""
        if self.is_loaded:
            BoxText("消息：正在卸载 DeepSeek-OCR 模型...")
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            self.is_loaded = False
            BoxText("成功：DeepSeek-OCR 模型已卸载")


# 全局单例
_dsocr_instance = None

def get_dsocr_instance(model_name='deepseek-ai/DeepSeek-OCR', gpu_id='0'):
    """
    获取 DeepSeek-OCR 单例
    
    Args:
        model_name: 模型名称或本地路径
        gpu_id: GPU设备ID
        
    Returns:
        DeepSeekOCR: DeepSeek-OCR实例
    """
    global _dsocr_instance
    if _dsocr_instance is None:
        _dsocr_instance = DeepSeekOCR(model_name=model_name, gpu_id=gpu_id)
    return _dsocr_instance


def infer_poker_local(image_path, model_name='deepseek-ai/DeepSeek-OCR', gpu_id='0'):
    """
    全局函数 - 德州扑克本地推理（对外接口）
    
    Args:
        image_path: 输入图片路径
        model_name: 模型名称或本地路径
        gpu_id: GPU设备ID
        
    Returns:
        dict: 识别结果的JSON对象
    """
    dsocr = get_dsocr_instance(model_name=model_name, gpu_id=gpu_id)
    return dsocr.infer_poker(image_path)


# 测试代码
if __name__ == "__main__":
    # 测试本地推理
    test_image = "temp/real_img.png"
    if os.path.exists(test_image):
        result = infer_poker_local(test_image)
        if result:
            print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"测试图片不存在: {test_image}")

