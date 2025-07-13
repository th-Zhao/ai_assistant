import os
from typing import Optional

class Config:
    # API配置
    OPENAI_API_KEY = "sk-WiOaM4jnrFyVW6qkIf9h6NfcuB5vdxyVt38GWAB05L7LGvKA"
    OPENAI_BASE_URL = "https://api.openai-proxy.org/v1"
    OPENAI_MODEL = "gpt-4o-mini"
    
    DEEPSEEK_API_KEY = "sk-yjtmiuuqonhwahtggfwtexiwupirrvrygoecphwmgsfotmrp"
    DEEPSEEK_BASE_URL = "https://api.siliconflow.cn/v1"
    DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-R1"
    
    # Redis配置
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_DB = 0
    REDIS_PASSWORD = None  # 如果需要密码请设置
    REDIS_DECODE_RESPONSES = True
    REDIS_CONNECTION_TIMEOUT = 10
    
    # 会话配置
    SESSION_EXPIRE_TIME = 3600 * 24  # 24小时后过期
    MAX_CONVERSATION_HISTORY = 20    # 每个会话最多保留20轮对话
    CONTEXT_WINDOW_SIZE = 6          # 每次使用最近6轮对话作为上下文
    
    # 项目配置
    UPLOAD_FOLDER = "uploads"
    VECTOR_STORE_PATH = "vector_store"
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx'}
    
    # RAG配置
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RESULTS = 5
    
    # 服务配置
    HOST = "0.0.0.0"
    PORT = 8000
    DEBUG = True
    
    @classmethod
    def get_openai_config(cls):
        return {
            "model": cls.OPENAI_MODEL,
            "base_url": cls.OPENAI_BASE_URL,
            "api_key": cls.OPENAI_API_KEY,
            "temperature": 0.7
        }
    
    @classmethod
    def get_deepseek_config(cls):
        return {
            "model": cls.DEEPSEEK_MODEL,
            "api_key": cls.DEEPSEEK_API_KEY,
            "base_url": cls.DEEPSEEK_BASE_URL,
            "temperature": 0.3
        }
    
    @classmethod
    def get_redis_config(cls):
        return {
            "host": cls.REDIS_HOST,
            "port": cls.REDIS_PORT,
            "db": cls.REDIS_DB,
            "password": cls.REDIS_PASSWORD,
            "decode_responses": cls.REDIS_DECODE_RESPONSES,
            "socket_connect_timeout": cls.REDIS_CONNECTION_TIMEOUT,
            "socket_timeout": cls.REDIS_CONNECTION_TIMEOUT
        }

# 确保必要的目录存在
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.VECTOR_STORE_PATH, exist_ok=True) 