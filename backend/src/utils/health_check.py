"""
健康检查工具模块
"""

import os
import shutil
from typing import Any, Dict

import psutil

from ..config import settings
from ..constants import CPU_WARNING_THRESHOLD


def check_disk_space() -> Dict[str, Any]:
    """检查磁盘空间"""
    try:
        # 获取项目根目录所在磁盘的使用情况
        root_path = os.path.abspath(".")
        disk_usage = shutil.disk_usage(root_path)

        total_gb = disk_usage.total / (1024**3)
        free_gb = disk_usage.free / (1024**3)
        used_gb = disk_usage.used / (1024**3)
        used_percent = (disk_usage.used / disk_usage.total) * 100

        threshold_gb = settings.HEALTH_CHECK_DISK_THRESHOLD_GB
        is_healthy = free_gb >= threshold_gb

        return {
            "status": "healthy" if is_healthy else "warning",
            "total_gb": round(total_gb, 2),
            "free_gb": round(free_gb, 2),
            "used_gb": round(used_gb, 2),
            "used_percent": round(used_percent, 2),
            "threshold_gb": threshold_gb,
            "is_healthy": is_healthy,
        }
    except Exception as e:
        # 延迟导入logger避免循环导入
        from ..utils import logger

        logger.error(f"磁盘空间检查失败: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "is_healthy": False,
        }


def check_memory() -> Dict[str, Any]:
    """检查内存使用"""
    try:
        memory = psutil.virtual_memory()

        total_mb = memory.total / (1024**2)
        available_mb = memory.available / (1024**2)
        used_mb = memory.used / (1024**2)
        used_percent = memory.percent

        threshold_mb = settings.HEALTH_CHECK_MEMORY_THRESHOLD_MB
        is_healthy = available_mb >= threshold_mb

        return {
            "status": "healthy" if is_healthy else "warning",
            "total_mb": round(total_mb, 2),
            "available_mb": round(available_mb, 2),
            "used_mb": round(used_mb, 2),
            "used_percent": round(used_percent, 2),
            "threshold_mb": threshold_mb,
            "is_healthy": is_healthy,
        }
    except Exception as e:
        # 延迟导入logger避免循环导入
        from ..utils import logger

        logger.error(f"内存检查失败: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "is_healthy": False,
        }


def check_database() -> Dict[str, Any]:
    """检查数据库连接"""
    try:
        from ..database import get_db

        db = get_db()
        is_connected = db.check_connection()

        return {
            "status": "healthy" if is_connected else "error",
            "is_connected": is_connected,
            "is_healthy": is_connected,
        }
    except Exception as e:
        # 延迟导入logger避免循环导入
        from ..utils import logger

        logger.error(f"数据库检查失败: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "is_connected": False,
            "is_healthy": False,
        }


def check_cpu() -> Dict[str, Any]:
    """
    检查CPU使用率

    获取当前CPU使用率，超过阈值认为不健康。

    Returns:
        Dict[str, Any]: CPU检查结果，包含：
            - status: 健康状态（"healthy"或"warning"）
            - cpu_percent: CPU使用率百分比
            - cpu_count: CPU核心数
            - is_healthy: 是否健康（布尔值）
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()

        # CPU使用率超过阈值认为不健康
        is_healthy = cpu_percent < CPU_WARNING_THRESHOLD

        return {
            "status": "healthy" if is_healthy else "warning",
            "cpu_percent": round(cpu_percent, 2),
            "cpu_count": cpu_count,
            "is_healthy": is_healthy,
        }
    except Exception as e:
        # 延迟导入logger避免循环导入
        from ..utils import logger

        logger.error(f"CPU检查失败: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "is_healthy": False,
        }


def comprehensive_health_check() -> Dict[str, Any]:
    """综合健康检查"""
    disk = check_disk_space()
    memory = check_memory()
    database = check_database()
    cpu = check_cpu()

    # 判断整体健康状态
    all_healthy = (
        disk.get("is_healthy", False)
        and memory.get("is_healthy", False)
        and database.get("is_healthy", False)
        and cpu.get("is_healthy", False)
    )

    overall_status = "healthy" if all_healthy else "degraded"

    return {
        "status": overall_status,
        "checks": {
            "disk": disk,
            "memory": memory,
            "database": database,
            "cpu": cpu,
        },
        "all_healthy": all_healthy,
    }
