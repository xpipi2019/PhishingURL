"""
配置管理模块

从环境变量加载应用配置，支持通过.env文件或系统环境变量配置。
所有配置项都有合理的默认值，可以直接运行而无需配置。

配置优先级：
1. 环境变量
2. .env文件
3. 默认值

示例：
    from src.config import settings
    print(settings.APP_NAME)
"""

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# 加载.env文件（如果存在）
# 路径：backend目录/.env 或 项目根目录/.env
backend_env_path = Path(__file__).parent.parent / ".env"
root_env_path = Path(__file__).parent.parent.parent / ".env"
env_path = backend_env_path if backend_env_path.exists() else root_env_path
load_dotenv(dotenv_path=env_path)


class Settings:
    """
    应用配置类

    管理所有应用配置项，包括：
    - 应用基本信息
    - 服务器配置
    - 数据库配置
    - CORS配置
    - 日志配置
    - 功能参数配置
    """

    # ========== 应用基本信息 ==========
    APP_NAME: str = os.getenv("APP_NAME", "网络安全威胁检测系统")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # 服务器配置
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8088"))

    # 目录配置
    MODELS_DIR: str = os.getenv("MODELS_DIR", "models")
    IMAGES_DIR: str = os.getenv("IMAGES_DIR", "images")
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    LOGS_DIR: str = os.getenv("LOGS_DIR", "logs")
    DB_DIR: str = os.getenv("DB_DIR", "db")

    # 数据库配置
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///db/phishing_detection.db")

    # CORS配置
    CORS_ORIGINS: List[str] = os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:8080"
    ).split(",")
    CORS_ALLOW_CREDENTIALS: bool = (
        os.getenv("CORS_ALLOW_CREDENTIALS", "True").lower() == "true"
    )
    CORS_ALLOW_METHODS: List[str] = os.getenv(
        "CORS_ALLOW_METHODS", "GET,POST,PUT,DELETE,OPTIONS"
    ).split(",")
    CORS_ALLOW_HEADERS: List[str] = os.getenv("CORS_ALLOW_HEADERS", "*").split(",")

    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "json")  # json 或 text
    LOG_RETENTION_DAYS: int = int(os.getenv("LOG_RETENTION_DAYS", "30"))
    LOG_ERROR_RETENTION_DAYS: int = int(os.getenv("LOG_ERROR_RETENTION_DAYS", "90"))

    # 特征提取配置
    FEATURE_EXTRACTION_TIMEOUT: int = int(os.getenv("FEATURE_EXTRACTION_TIMEOUT", "10"))
    DEFAULT_FEATURES_COUNT: int = int(os.getenv("DEFAULT_FEATURES_COUNT", "30"))

    # 批量预测配置
    BATCH_PREDICT_MAX_SIZE: int = int(os.getenv("BATCH_PREDICT_MAX_SIZE", "100"))
    BATCH_PREDICT_TIMEOUT: int = int(os.getenv("BATCH_PREDICT_TIMEOUT", "300"))

    # 健康检查配置
    HEALTH_CHECK_DISK_THRESHOLD_GB: float = float(
        os.getenv("HEALTH_CHECK_DISK_THRESHOLD_GB", "1.0")
    )
    HEALTH_CHECK_MEMORY_THRESHOLD_MB: float = float(
        os.getenv("HEALTH_CHECK_MEMORY_THRESHOLD_MB", "100")
    )

    @classmethod
    def ensure_directories(cls):
        """
        确保所有必要的目录存在

        在应用启动时自动创建以下目录（如果不存在）：
        - models/ : 模型文件存储目录
        - images/ : 可视化图片存储目录
        - data/ : 数据文件目录
        - logs/ : 日志文件目录
        - db/ : 数据库文件目录
        """
        directories = [
            cls.MODELS_DIR,
            cls.IMAGES_DIR,
            cls.DATA_DIR,
            cls.LOGS_DIR,
            cls.DB_DIR,
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# 创建全局配置实例
# 在模块导入时自动初始化，确保目录存在
settings = Settings()
settings.ensure_directories()
