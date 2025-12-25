"""
模型信息管理模块

管理模型信息、版本控制、最佳模型追踪等功能。
同时维护JSON文件和数据库两种存储方式（向后兼容）。
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from ..constants import DEFAULT_FEATURES_COUNT
from ..database import get_db
from ..utils import logger


class ModelManager:
    """
    模型管理器

    负责管理所有训练好的模型信息，包括：
    - 模型元数据存储（JSON + 数据库）
    - 最佳模型追踪
    - 模型状态查询
    - 系统健康信息

    Attributes:
        models_dir: 模型文件保存目录
        images_dir: 图片文件保存目录
        data_dir: 数据文件目录
        best_model_name: 最佳模型类型名称
        best_accuracy: 最佳模型准确率
        model_info: 模型信息字典（从JSON加载）
        features_count: 特征数量
        start_time: 系统启动时间（用于计算运行时长）
    """

    def __init__(
        self,
        models_dir: str = "models",
        images_dir: str = "images",
        data_dir: str = "data",
    ) -> None:
        """
        初始化模型管理器

        Args:
            models_dir: 模型保存目录，默认为 "models"
            images_dir: 图片保存目录，默认为 "images"
            data_dir: 数据文件目录，默认为 "data"

        Note:
            会自动创建必要的目录，并从JSON文件加载已有模型信息
        """
        self.models_dir: str = models_dir
        self.images_dir: str = images_dir
        self.data_dir: str = data_dir
        self.best_model_name: Optional[str] = None
        self.best_accuracy: float = 0.0
        self.model_info: Dict[str, Dict[str, Any]] = {}  # 存储模型信息
        self.features_count: Optional[int] = None  # 特征数量
        self.start_time: float = time.time()

        # 创建必要的目录
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        logger.info(
            f"初始化模型管理器: models_dir={models_dir}, images_dir={images_dir}"
        )
        # 加载模型信息（如果存在）
        self._load_model_info()

    def _load_model_info(self) -> None:
        """
        从JSON文件加载模型信息

        从 models/model_info.json 文件加载已保存的模型信息，
        包括模型列表、最佳模型、特征数量等。

        Note:
            如果文件不存在或加载失败，会使用空字典初始化
        """
        info_file = os.path.join(self.models_dir, "model_info.json")
        if os.path.exists(info_file):
            try:
                with open(info_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.model_info = data.get("model_info", {})
                    self.best_model_name = data.get("best_model_name")
                    self.best_accuracy = data.get("best_accuracy", 0.0)
                    # 加载特征数量（从最佳模型或第一个可用模型中获取）
                    if self.best_model_name and self.best_model_name in self.model_info:
                        self.features_count = self.model_info[self.best_model_name].get(
                            "features_count"
                        )
                    elif self.model_info:
                        # 如果没有最佳模型，从第一个模型中获取
                        first_model = next(iter(self.model_info.values()))
                        self.features_count = first_model.get("features_count")
                logger.info(
                    f"模型信息加载成功: 共 {len(self.model_info)} 个模型, 最佳模型: {self.best_model_name}"
                )
            except Exception as e:
                logger.error(f"加载模型信息失败: {e}")

    def _save_model_info(self) -> None:
        """
        保存模型信息到JSON文件

        将当前模型信息保存到 models/model_info.json 文件。
        用于向后兼容，数据库是主要存储方式。

        Raises:
            Exception: 当文件写入失败时
        """
        info_file = os.path.join(self.models_dir, "model_info.json")
        data = {
            "model_info": self.model_info,
            "best_model_name": self.best_model_name,
            "best_accuracy": self.best_accuracy,
            "updated_at": datetime.now().isoformat(),
        }
        try:
            with open(info_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"模型信息保存成功: {info_file}")
        except Exception as e:
            logger.error(f"保存模型信息失败: {info_file}, 错误: {str(e)}")
            raise

    def save_model_info(
        self,
        model_type: str,
        model_path: str,
        image_path: str,
        additional_images: Dict[str, str],
        accuracy: float,
        confusion_matrix: list,
        classification_report: Dict[str, Any],
        training_time: float,
        train_samples: int,
        test_samples: int,
        features_count: int,
        data_cleaning: Dict[str, int],
    ):
        """
        保存模型信息（同时保存到JSON和数据库）

        Args:
            model_type: 模型类型
            model_path: 模型文件路径
            image_path: 混淆矩阵图片路径
            additional_images: 额外图片路径字典
            accuracy: 准确率
            confusion_matrix: 混淆矩阵
            classification_report: 分类报告
            training_time: 训练时间
            train_samples: 训练样本数
            test_samples: 测试样本数
            features_count: 特征数量
            data_cleaning: 数据清洗信息
        """
        # 生成版本号（使用时间戳）
        version = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_info = {
            "model_type": model_type,
            "model_path": model_path,
            "image_path": image_path,
            "additional_images": additional_images,
            "accuracy": float(accuracy),
            "confusion_matrix": confusion_matrix,
            "classification_report": classification_report,
            "training_time": training_time,
            "train_samples": train_samples,
            "test_samples": test_samples,
            "features_count": features_count,
            "data_cleaning": data_cleaning,
            "created_at": datetime.now().isoformat(),
            "version": version,  # 添加版本号
        }

        self.model_info[model_type] = model_info
        self.features_count = features_count  # 更新特征数量

        # 判断是否是最佳模型
        is_best = accuracy > self.best_accuracy
        if is_best:
            self.best_accuracy = accuracy
            self.best_model_name = model_type

        # 保存到JSON文件（向后兼容）
        self._save_model_info()

        # 保存到数据库
        try:
            db = get_db()
            db.create_model_version(
                model_type=model_type,
                version=version,
                model_path=model_path,
                image_path=image_path,
                additional_images=additional_images,
                accuracy=accuracy,
                confusion_matrix=confusion_matrix,
                classification_report=classification_report,
                training_time=training_time,
                train_samples=train_samples,
                test_samples=test_samples,
                features_count=features_count,
                data_cleaning=data_cleaning,
                is_best=is_best,
            )
            logger.info(f"模型版本已保存到数据库: {model_type} v{version}")
        except Exception as e:
            logger.error(f"保存模型版本到数据库失败: {str(e)}")
            # 不抛出异常，允许继续使用JSON方式

    def get_model_status(self, model_type: str) -> Dict[str, Any]:
        """
        获取模型状态

        Args:
            model_type: 模型类型

        Returns:
            模型状态字典
        """
        if model_type not in self.model_info:
            raise ValueError(f"模型 {model_type} 不存在")

        info = self.model_info[model_type].copy()
        info["is_best_model"] = model_type == self.best_model_name
        info["model_exists"] = os.path.exists(info["model_path"])
        info["image_exists"] = os.path.exists(info["image_path"])

        return info

    def get_model_info(self, model_type: Optional[str] = None) -> Dict[str, Any]:
        """
        获取模型信息

        Args:
            model_type: 模型类型，如果为None则返回最佳模型信息

        Returns:
            模型信息字典
        """
        if model_type is None:
            model_type = self.best_model_name

        if model_type is None:
            logger.error("没有可用的模型，请先训练模型")
            raise ValueError("没有可用的模型，请先训练模型")

        if model_type not in self.model_info:
            logger.error(f"模型不存在: {model_type}")
            raise ValueError(f"模型 {model_type} 不存在")

        return self.model_info[model_type]

    def get_health_info(self) -> Dict[str, Any]:
        """
        获取系统健康信息

        检查模型状态、系统运行时长等基础健康信息。
        注意：此方法只检查模型相关健康，完整健康检查请使用 health_check 模块。

        Returns:
            Dict[str, Any]: 健康信息字典，包含：
                - status: 健康状态（"healthy" 或 "degraded"）
                - has_trained_models: 是否有已训练的模型
                - model_count: 模型数量
                - best_model: 最佳模型类型
                - best_accuracy: 最佳模型准确率
                - can_predict: 是否可以预测
                - model_working: 模型是否可以正常工作
                - uptime_seconds: 运行时长（秒）
                - uptime_formatted: 格式化的运行时长
        """
        has_models = len(self.model_info) > 0
        can_predict = self.best_model_name is not None

        # 测试模型是否可以正常工作
        model_working = False
        if can_predict:
            try:
                from .model_predictor import ModelPredictor

                model_info = self.get_model_info()
                predictor = ModelPredictor(model_info)
                predictor.load_model()

                # 创建一个测试特征向量（全0）
                features_count = self.features_count
                if features_count is None:
                    features_count = DEFAULT_FEATURES_COUNT

                test_features = np.zeros(features_count)
                predictor.predict(test_features)
                model_working = True
            except Exception:
                pass

        uptime = time.time() - self.start_time

        return {
            "status": "healthy"
            if (has_models and can_predict and model_working)
            else "degraded",
            "has_trained_models": has_models,
            "model_count": len(self.model_info),
            "best_model": self.best_model_name,
            "best_accuracy": self.best_accuracy,
            "can_predict": can_predict,
            "model_working": model_working,
            "uptime_seconds": uptime,
            "uptime_formatted": self._format_uptime(uptime),
        }

    def _format_uptime(self, seconds: float) -> str:
        """
        格式化运行时间

        将秒数转换为易读的时间格式，如 "1天2小时30分钟15秒"。

        Args:
            seconds: 运行时长（秒）

        Returns:
            str: 格式化的时间字符串，如 "1天2小时30分钟15秒"
        """
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        parts = []
        if days > 0:
            parts.append(f"{days}天")
        if hours > 0:
            parts.append(f"{hours}小时")
        if minutes > 0:
            parts.append(f"{minutes}分钟")
        if secs > 0 or not parts:
            parts.append(f"{secs}秒")

        return "".join(parts)

    def get_all_models_info(self) -> Dict[str, Any]:
        """
        获取所有模型的信息

        返回所有已训练模型的详细信息，按准确率降序排列。

        Returns:
            Dict[str, Any]: 所有模型信息字典，包含：
                - total_models: 总模型数
                - best_model: 最佳模型类型
                - best_accuracy: 最佳模型准确率
                - models: 模型信息列表（按准确率排序）
                - updated_at: 更新时间（ISO格式）
        """
        # 准备所有模型信息
        models_data = []
        for model_type, info in self.model_info.items():
            model_data = info.copy()
            model_data["is_best_model"] = model_type == self.best_model_name
            model_data["model_exists"] = os.path.exists(model_data["model_path"])
            model_data["image_exists"] = os.path.exists(model_data["image_path"])
            models_data.append(model_data)

        # 按准确率排序
        models_data.sort(key=lambda x: x["accuracy"], reverse=True)

        return {
            "total_models": len(self.model_info),
            "best_model": self.best_model_name,
            "best_accuracy": self.best_accuracy,
            "models": models_data,
            "updated_at": datetime.now().isoformat(),
        }
