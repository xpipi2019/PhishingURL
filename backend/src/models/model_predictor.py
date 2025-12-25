"""
模型预测模块

提供模型加载、单样本预测和批量预测功能。
"""

from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..data import FeatureExtractor
from ..utils import logger


class ModelPredictor:
    """
    模型预测器

    负责加载训练好的模型，对URL或特征向量进行预测。
    支持自动特征提取和批量预测。

    Attributes:
        model_info: 模型信息字典，包含模型路径、特征数量等
        _classifier: 模型分类器实例（延迟加载）
        _scaler: 特征标准化器（延迟加载）
        _label_encoder: 标签编码器（XGBoost需要，延迟加载）
        feature_extractor: URL特征提取器实例
    """

    def __init__(self, model_info: Dict[str, Any]) -> None:
        """
        初始化模型预测器

        Args:
            model_info: 模型信息字典，必须包含以下键：
                - model_path: 模型文件路径
                - features_count: 特征数量（可选）

        Note:
            模型文件会在首次预测时自动加载（延迟加载）
        """
        self.model_info: Dict[str, Any] = model_info
        self._classifier: Optional[BaseEstimator] = None
        self._scaler: Optional[StandardScaler] = None
        self._label_encoder: Optional[LabelEncoder] = None
        self.feature_extractor: FeatureExtractor = FeatureExtractor()

    def load_model(self) -> None:
        """
        加载模型文件

        从磁盘加载训练好的模型、标准化器和标签编码器。
        模型文件应是通过joblib保存的字典，包含：
        - classifier: 训练好的分类器
        - scaler: 特征标准化器
        - label_encoder: 标签编码器（可选，XGBoost需要）

        Raises:
            FileNotFoundError: 当模型文件不存在时
            Exception: 当模型文件损坏或加载失败时

        Note:
            如果模型已加载，此方法不会重复加载
        """
        model_path = self.model_info["model_path"]
        logger.debug(f"加载模型: {model_path}")
        try:
            model_data = joblib.load(model_path)
            self._classifier = model_data["classifier"]
            self._scaler = model_data["scaler"]
            self._label_encoder = model_data.get("label_encoder")
            logger.info(f"模型加载成功: {model_path}")
        except FileNotFoundError:
            logger.error(f"模型文件不存在: {model_path}")
            raise
        except Exception as e:
            logger.error(f"模型加载失败: {model_path}, 错误: {str(e)}")
            raise

    def predict(
        self, features: Union[np.ndarray, str], model_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        预测单个样本

        支持两种输入方式：
        1. 特征向量（numpy数组）：直接使用提供的特征
        2. URL字符串：自动从URL提取特征

        Args:
            features:
                - np.ndarray: 特征向量，形状为 (n_features,) 或 (1, n_features)
                - str: URL字符串，会自动提取特征
            model_type: 模型类型（可选，用于验证，不影响预测）

        Returns:
            Dict[str, Any]: 预测结果字典，包含：
                - prediction: 预测标签（字符串）
                - probabilities: 预测概率字典，键为类别标签，值为概率

        Raises:
            ValueError: 当特征数量不匹配或模型未加载时
            Exception: 当特征提取失败时

        Examples:
            >>> predictor = ModelPredictor(model_info)
            >>> # 使用特征向量
            >>> result = predictor.predict(np.array([1, 0, -1, ...]))
            >>> # 使用URL
            >>> result = predictor.predict("https://example.com")
        """
        if self._classifier is None:
            self.load_model()

        # 确保模型已加载
        if self._classifier is None or self._scaler is None:
            raise ValueError("模型或缩放器未加载")

        # 如果传入的是URL字符串，自动提取特征
        if isinstance(features, str):
            url = features
            logger.info(f"从URL提取特征: {url}")
            features = self.feature_extractor.extract_url_features(url)
            logger.info(f"特征提取完成，特征数量: {len(features)}")

        # 验证特征数量
        features = np.array(features).flatten()
        expected_features = getattr(self._scaler, "n_features_in_", None)
        if expected_features is None:
            expected_features = self.model_info.get("features_count")

        if expected_features is not None and len(features) != expected_features:
            logger.error(
                f"特征数量不匹配: 期望 {expected_features} 个特征，但得到 {len(features)} 个特征"
            )
            raise ValueError(
                f"特征数量不匹配: 期望 {expected_features} 个特征，但得到 {len(features)} 个特征"
            )

        logger.debug(f"开始预测，特征数量: {len(features)}")

        # 特征缩放
        features_scaled = self._scaler.transform(features.reshape(1, -1))

        # 预测
        prediction = self._classifier.predict(features_scaled)[0]  # type: ignore

        # 如果是XGBoost，需要解码标签
        if self._label_encoder is not None:
            prediction_label = self._label_encoder.inverse_transform([prediction])[0]
        else:
            prediction_label = prediction

        # 获取预测概率（如果支持）
        prob_dict = {}
        try:
            probabilities = self._classifier.predict_proba(features_scaled)[0]  # type: ignore
            if self._label_encoder is not None and hasattr(
                self._label_encoder, "classes_"
            ):
                classes = self._label_encoder.classes_  # type: ignore
                for i, label in enumerate(classes):  # type: ignore
                    prob_dict[str(label)] = float(probabilities[i])
            else:
                for i, prob in enumerate(probabilities):
                    prob_dict[str(i)] = float(prob)
        except AttributeError:
            # 某些模型（如SVM）可能不支持predict_proba
            logger.warning("模型不支持predict_proba方法")
            pass
        except Exception as e:
            # 其他异常，记录但不中断流程
            logger.warning(f"获取预测概率失败: {e}")

        logger.info(f"预测完成: prediction={prediction_label}")
        return {
            "prediction": str(prediction_label),
            "probabilities": prob_dict,
        }

    def predict_batch(self, features_list: list) -> Dict[str, Any]:
        """
        批量预测

        对多个特征向量进行批量预测，比单次预测更高效。

        Args:
            features_list: 特征列表，每个元素是一个特征向量（列表或numpy数组）
                形状应为 (n_samples, n_features)

        Returns:
            Dict[str, Any]: 批量预测结果字典，包含：
                - results: 预测结果列表，每个元素包含：
                    - index: 样本索引
                    - prediction: 预测标签
                    - probabilities: 预测概率字典
                - total: 总预测数量

        Raises:
            ValueError: 当特征数量不匹配或输入格式错误时

        Examples:
            >>> features_list = [
            ...     [1, 0, -1, ...],  # 样本1
            ...     [0, 1, 1, ...],   # 样本2
            ... ]
            >>> result = predictor.predict_batch(features_list)
            >>> print(f"预测了 {result['total']} 个样本")
        """
        if self._classifier is None:
            self.load_model()

        # 确保模型已加载
        if self._classifier is None or self._scaler is None:
            raise ValueError("模型或缩放器未加载")

        # 转换为numpy数组
        features_array = np.array(features_list)

        # 验证特征数量
        if len(features_array.shape) != 2:
            raise ValueError(
                f"特征列表应该是二维数组，但得到形状: {features_array.shape}"
            )

        expected_features = getattr(self._scaler, "n_features_in_", None)
        if expected_features is None:
            expected_features = self.model_info.get("features_count")

        if (
            expected_features is not None
            and features_array.shape[1] != expected_features
        ):
            logger.error(
                f"批量预测特征数量不匹配: 期望 {expected_features} 个特征，但得到 {features_array.shape[1]} 个特征"
            )
            raise ValueError(
                f"特征数量不匹配: 期望 {expected_features} 个特征，但得到 {features_array.shape[1]} 个特征"
            )

        logger.debug(f"开始批量预测，样本数: {len(features_list)}")

        # 特征缩放
        features_scaled = self._scaler.transform(features_array)

        # 预测
        predictions = self._classifier.predict(features_scaled)  # type: ignore

        # 如果是XGBoost，需要解码标签
        if self._label_encoder is not None:
            prediction_labels = self._label_encoder.inverse_transform(predictions)
        else:
            prediction_labels = predictions

        # 获取预测概率（如果支持）
        results = []
        try:
            probabilities = self._classifier.predict_proba(features_scaled)  # type: ignore[attr-defined]
            for i, (pred, probs) in enumerate(zip(prediction_labels, probabilities)):
                prob_dict = {}
                if self._label_encoder is not None and hasattr(
                    self._label_encoder, "classes_"
                ):
                    classes = self._label_encoder.classes_  # type: ignore
                    for j, label in enumerate(classes):  # type: ignore
                        prob_dict[str(label)] = float(probs[j])
                else:
                    for j, prob in enumerate(probs):
                        prob_dict[str(j)] = float(prob)

                results.append(
                    {"index": i, "prediction": str(pred), "probabilities": prob_dict}
                )
        except AttributeError:
            # 某些模型（如SVM）可能不支持predict_proba
            logger.warning("模型不支持predict_proba方法，批量预测")
            for i, pred in enumerate(prediction_labels):
                results.append(
                    {"index": i, "prediction": str(pred), "probabilities": {}}
                )
        except Exception as e:
            # 其他异常，记录但不中断流程
            logger.warning(f"获取预测概率失败: {e}")
            for i, pred in enumerate(prediction_labels):
                results.append(
                    {"index": i, "prediction": str(pred), "probabilities": {}}
                )

        logger.info(f"批量预测完成: 共 {len(results)} 个结果")
        return {
            "results": results,
            "total": len(results),
        }
