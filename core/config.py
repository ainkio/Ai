"""
core/config.py - 集中式配置管理模块

将项目中所有模型URL、API密钥、模型名称、服务器配置等统一归口到此文件。
其他模块通过 `from core.config import *` 获取配置常量。

配置来源优先级：
  1. 环境变量（.env 文件或系统环境变量）
  2. model/config.ini 配置文件
  3. 代码中的默认值
"""

import os
import configparser

# ============================================================
# 加载 .env 文件（如果存在）
# ============================================================
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_PROJECT_ROOT, '.env'))
except ImportError:
    pass  # python-dotenv 未安装时，依赖系统环境变量

# ============================================================
# 配置文件路径
# ============================================================
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, 'model', 'config.ini')


def _load_config():
    """加载 model/config.ini 配置文件"""
    config = configparser.ConfigParser(interpolation=None)
    config.read(_CONFIG_PATH, encoding='utf-8')
    return config


_cfg = _load_config()


def _get(section, key, fallback=''):
    """从配置文件读取值（带默认值）"""
    return _cfg.get(section, key, fallback=fallback)


# ============================================================
# API 密钥（环境变量优先，回退到配置文件）
# ============================================================
ALI_API_KEY = os.getenv('ALI_API_KEY', _get('api_keys', 'ali_api_key'))
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', _get('api_keys', 'deepseek_api_key'))
XAI_API_KEY = os.getenv('XAI_API_KEY', _get('api_keys', 'xai_api_key'))

# ============================================================
# API 基础 URL
# ============================================================
ALI_BASE_URL = _get('api_urls', 'ali_base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
DEEPSEEK_BASE_URL = _get('api_urls', 'deepseek_base_url', 'https://api.deepseek.com')
XAI_BASE_URL = _get('api_urls', 'xai_base_url', 'https://api.x.ai/v1')

# ============================================================
# 视觉模型名称（VLM - 用于牌面/数字识别）
# ============================================================
ALI_VISION_MODEL = _get('vision_models', 'ali_vision_model', 'qwen3-vl-plus')
ALI_VISION_FLASH_MODEL = _get('vision_models', 'ali_vision_flash_model', 'qwen3-vl-flash')

# ============================================================
# 语言模型名称（LLM - 用于策略推理决策）
# ============================================================
ALI_LLM_MODEL = _get('language_models', 'ali_model', 'qwen-max-latest')
DEEPSEEK_LLM_MODEL = _get('language_models', 'deepseek_model', 'deepseek-chat')
XAI_LLM_MODEL = _get('language_models', 'xai_model', 'grok-3')

# ============================================================
# 决策引擎配置
# ============================================================
RULE = _cfg.getfloat('decision', 'rule', fallback=15)
PREFILTER_ENABLED = _cfg.getboolean('decision', 'prefilter_enabled', fallback=True)
CORRECTION_ENABLED = _cfg.getboolean('decision', 'correction_enabled', fallback=False)
AIPA = _get('decision', 'aipa', 'dddd')
IDEN = _get('decision', 'iden', '')

# 兼容旧代码的别名（amd/dmd/xmd 对应各引擎的模型名）
AMD = ALI_LLM_MODEL
DMD = DEEPSEEK_LLM_MODEL
XMD = XAI_LLM_MODEL

# ============================================================
# 服务器配置
# ============================================================
POKER_SERVER_DOMAIN = os.getenv('POKER_SERVER_DOMAIN', _get('server', 'poker_server_domain', 'localhost'))
POKER_AUTH_URL = os.getenv('POKER_AUTH_URL', _get('server', 'poker_auth_url', ''))
GLOBAL_DOMAIN = f'http://{POKER_SERVER_DOMAIN}'

# 仅用于兼容旧代码中 only_domain 变量
only_domain = POKER_SERVER_DOMAIN
global_domain = GLOBAL_DOMAIN

# ============================================================
# 数据库配置（环境变量优先）
# ============================================================
DB_HOST = os.getenv('DB_HOST', _get('server', 'db_host', 'localhost'))
DB_USER = os.getenv('DB_USER', _get('server', 'db_user', 'root'))
DB_PASSWORD = os.getenv('DB_PASSWORD', _get('server', 'db_password', ''))
DB_NAME = os.getenv('DB_NAME', _get('server', 'db_name', 'poker'))

# ============================================================
# HuggingFace 镜像（中国大陆用户加速下载模型）
# ============================================================
HF_ENDPOINT = _get('misc', 'hf_endpoint', 'https://hf-mirror.com')
os.environ['HF_ENDPOINT'] = HF_ENDPOINT
os.environ['CURL_CA_BUNDLE'] = ''


def Read_Config(type="mode"):
    """
    兼容旧接口：读取模型配置
    返回: (amd, dmd, xmd, aipa, iden, rule, prefilter_enabled, correction_enabled)
    """
    if type == "mode":
        return (AMD, DMD, XMD, AIPA, IDEN, str(int(RULE)),
                PREFILTER_ENABLED, CORRECTION_ENABLED)
    return None
