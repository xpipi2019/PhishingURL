"""
日志工具模块

使用loguru实现统一的日志管理，支持JSON和文本两种格式。

功能特性：
- 支持JSON和文本两种日志格式
- 自动日志轮转（按天）
- 日志压缩（zip格式）
- 错误日志单独记录
- 可配置的日志保留时间
- 控制台和文件双重输出

配置说明：
- LOG_FORMAT: "json" 或 "text"（在.env中配置）
- LOG_LEVEL: 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）
- LOG_RETENTION_DAYS: 应用日志保留天数（默认30天）
- LOG_ERROR_RETENTION_DAYS: 错误日志保留天数（默认90天）

日志文件位置：
- 应用日志: logs/app_YYYY-MM-DD.log
- 错误日志: logs/error_YYYY-MM-DD.log
"""

import json
import sys
from pathlib import Path

from loguru import logger

from ..config import settings


def json_sink(message):
    """JSON格式的sink函数 - 用于控制台输出"""
    record = message.record
    log_entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
    }

    # 添加异常信息
    if record["exception"] is not None:
        log_entry["exception"] = {
            "type": record["exception"].type.__name__,
            "value": str(record["exception"].value),
            "traceback": record["exception"].traceback.format_exc(),
        }

    # 添加额外的上下文信息
    if record.get("extra"):
        log_entry.update(record["extra"])

    print(json.dumps(log_entry, ensure_ascii=False), file=sys.stdout)


def json_file_sink(message):
    """JSON格式的文件sink函数"""
    record = message.record
    log_entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
    }

    # 添加异常信息
    if record["exception"] is not None:
        log_entry["exception"] = {
            "type": record["exception"].type.__name__,
            "value": str(record["exception"].value),
            "traceback": record["exception"].traceback.format_exc(),
        }

    # 添加额外的上下文信息
    if record.get("extra"):
        log_entry.update(record["extra"])

    # 返回JSON字符串，loguru会写入文件
    return json.dumps(log_entry, ensure_ascii=False) + "\n"


# 日志目录
LOG_DIR = Path(settings.LOGS_DIR)
LOG_DIR.mkdir(exist_ok=True)

# 移除默认的handler
logger.remove()

# 根据配置选择日志格式
log_format = settings.LOG_FORMAT.lower()
use_json = log_format == "json"

if use_json:
    # JSON格式 - 控制台输出（INFO级别及以上）
    logger.add(
        json_sink,
        level=settings.LOG_LEVEL,
        colorize=False,
    )

    # JSON格式 - 文件输出（DEBUG级别及以上）
    # 使用自定义sink函数，文件路径通过sink函数内部处理
    def app_file_sink(message):
        """应用日志文件sink"""
        log_str = json_file_sink(message)
        log_file = LOG_DIR / f"app_{message.record['time'].strftime('%Y-%m-%d')}.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_str)

    logger.add(
        app_file_sink,
        level="DEBUG",
        colorize=False,
    )

    # JSON格式 - 错误日志文件
    def error_file_sink(message):
        """错误日志文件sink"""
        log_str = json_file_sink(message)
        log_file = LOG_DIR / f"error_{message.record['time'].strftime('%Y-%m-%d')}.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_str)

    logger.add(
        error_file_sink,
        level="ERROR",
        colorize=False,
    )
else:
    # 文本格式 - 控制台输出（INFO级别及以上）
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.LOG_LEVEL,
        colorize=True,
    )

    # 文本格式 - 文件输出（DEBUG级别及以上）
    logger.add(
        LOG_DIR / "app_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="00:00",
        retention=f"{settings.LOG_RETENTION_DAYS} days",
        compression="zip",
        encoding="utf-8",
    )

    # 文本格式 - 错误日志文件
    logger.add(
        LOG_DIR / "error_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="00:00",
        retention=f"{settings.LOG_ERROR_RETENTION_DAYS} days",
        compression="zip",
        encoding="utf-8",
    )

# 导出logger实例
__all__ = ["logger"]
