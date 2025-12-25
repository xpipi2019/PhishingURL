"""
模型训练模块

提供多种机器学习模型的训练功能，包括：
- 数据加载和预处理
- 模型训练和保存
- 训练结果返回
"""

import os
from datetime import datetime
from typing import Any, Dict, Type

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from ..constants import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    KNN_DEFAULT_METRIC,
    KNN_DEFAULT_NEIGHBORS,
    KNN_DEFAULT_P,
    MODEL_FILE_EXTENSION,
    RANDOM_FOREST_DEFAULT_ESTIMATORS,
    TIMESTAMP_FORMAT,
)
from ..data import DataLoader, DataPreprocessor
from ..utils import logger


class ModelTrainer:
    """
    模型训练器

    负责加载数据、预处理、训练模型并保存训练结果。
    支持多种机器学习算法，包括分类和集成方法。

    Attributes:
        models_dir: 模型文件保存目录
        data_dir: 数据文件目录
        processed_dir: 处理后的数据保存目录
        data_loader: 数据加载器实例
        preprocessor: 数据预处理器实例
    """

    # 支持的模型类型映射
    MODEL_TYPES: Dict[str, Type[BaseEstimator]] = {
        "logistic_regression": LogisticRegression,
        "knn": KNeighborsClassifier,
        "svm": SVC,
        "kernel_svm": SVC,
        "naive_bayes": GaussianNB,
        "decision_tree": DecisionTreeClassifier,
        "random_forest": RandomForestClassifier,
        "xgboost": XGBClassifier,
    }

    def __init__(
        self,
        models_dir: str = "models",
        data_dir: str = "data",
    ) -> None:
        """
        初始化模型训练器

        Args:
            models_dir: 模型保存目录，默认为 "models"
            data_dir: 数据文件目录，默认为 "data"

        Note:
            会自动创建必要的目录结构
        """
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.processed_dir = os.path.join(data_dir, "processed")

        # 创建必要的目录
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        logger.info(f"初始化模型训练器: models_dir={models_dir}, data_dir={data_dir}")
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()

    def _get_model_classifier(self, model_type: str, **kwargs: Any) -> BaseEstimator:
        """
        获取模型分类器实例

        根据模型类型创建并配置相应的分类器，设置默认参数。

        Args:
            model_type: 模型类型，必须是 MODEL_TYPES 中支持的类型
            **kwargs: 模型额外参数，会覆盖默认参数

        Returns:
            BaseEstimator: 配置好的模型分类器实例

        Raises:
            ValueError: 当模型类型不支持时抛出异常

        Examples:
            >>> trainer = ModelTrainer()
            >>> classifier = trainer._get_model_classifier("logistic_regression")
            >>> classifier = trainer._get_model_classifier("knn", n_neighbors=10)
        """
        if model_type not in self.MODEL_TYPES:
            logger.error(f"不支持的模型类型: {model_type}")
            raise ValueError(
                f"不支持的模型类型: {model_type}。"
                f"支持的模型类型: {list(self.MODEL_TYPES.keys())}"
            )

        logger.debug(f"创建模型分类器: {model_type}")

        model_class = self.MODEL_TYPES[model_type]

        # 根据模型类型设置默认参数
        # 注意：类型检查器可能无法识别所有模型参数，使用 type: ignore 忽略
        if model_type == "logistic_regression":
            return model_class(random_state=DEFAULT_RANDOM_STATE, **kwargs)  # type: ignore[misc]
        elif model_type == "knn":
            return model_class(  # type: ignore[misc]
                n_neighbors=KNN_DEFAULT_NEIGHBORS,  # type: ignore[arg-type]
                metric=KNN_DEFAULT_METRIC,  # type: ignore[arg-type]
                p=KNN_DEFAULT_P,  # type: ignore[arg-type]
                **kwargs,
            )
        elif model_type == "svm":
            return model_class(  # type: ignore[misc]
                kernel="linear",  # type: ignore[arg-type]
                random_state=DEFAULT_RANDOM_STATE,  # type: ignore[arg-type]
                **kwargs,
            )
        elif model_type == "kernel_svm":
            return model_class(  # type: ignore[misc]
                kernel="rbf",  # type: ignore[arg-type]
                random_state=DEFAULT_RANDOM_STATE,  # type: ignore[arg-type]
                **kwargs,
            )
        elif model_type == "naive_bayes":
            return model_class(**kwargs)  # type: ignore[misc]
        elif model_type == "decision_tree":
            return model_class(  # type: ignore[misc]
                criterion="entropy",  # type: ignore[arg-type]
                random_state=DEFAULT_RANDOM_STATE,  # type: ignore[arg-type]
                **kwargs,
            )
        elif model_type == "random_forest":
            return model_class(  # type: ignore[misc]
                n_estimators=RANDOM_FOREST_DEFAULT_ESTIMATORS,  # type: ignore[arg-type]
                criterion="entropy",  # type: ignore[arg-type]
                random_state=DEFAULT_RANDOM_STATE,  # type: ignore[arg-type]
                **kwargs,
            )
        elif model_type == "xgboost":
            return model_class(**kwargs)  # type: ignore[misc]
        else:
            return model_class(**kwargs)  # type: ignore[misc]

    def train(
        self,
        dataset_path: str,
        model_type: str,
        test_size: float = DEFAULT_TEST_SIZE,
        random_state: int = DEFAULT_RANDOM_STATE,
        **model_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        训练机器学习模型

        完整的模型训练流程：
        1. 加载和清洗数据
        2. 分割训练集和测试集
        3. 特征缩放和标签编码
        4. 训练模型
        5. 保存模型文件
        6. 返回训练结果

        Args:
            dataset_path: 数据集CSV文件路径
            model_type: 模型类型，必须是 MODEL_TYPES 中支持的类型
            test_size: 测试集比例，范围 [0.1, 0.5]，默认 0.25
            random_state: 随机种子，用于保证结果可复现，默认 0
            **model_kwargs: 模型额外参数，会传递给模型构造函数

        Returns:
            Dict[str, Any]: 训练结果字典，包含以下键：
                - classifier: 训练好的模型分类器
                - preprocessor: 数据预处理器（包含scaler和label_encoder）
                - model_path: 保存的模型文件路径
                - model_type: 模型类型
                - features_count: 特征数量
                - X_train, X_test: 训练集和测试集特征
                - y_train, y_test: 训练集和测试集标签
                - X_train_scaled, X_test_scaled: 缩放后的特征
                - y_train_encoded: 编码后的训练标签
                - use_encoder: 是否使用了标签编码
                - data_cleaning: 数据清洗统计信息

        Raises:
            FileNotFoundError: 当数据集文件不存在时
            ValueError: 当模型类型不支持或参数无效时
            Exception: 当模型训练或保存失败时

        Examples:
            >>> trainer = ModelTrainer()
            >>> result = trainer.train(
            ...     dataset_path="data/PhishingData.csv",
            ...     model_type="logistic_regression",
            ...     test_size=0.25
            ... )
            >>> print(f"模型已保存到: {result['model_path']}")
        """
        logger.info(
            f"开始训练模型: model_type={model_type}, dataset_path={dataset_path}, test_size={test_size}, random_state={random_state}"
        )

        # 加载数据集
        dataset = self.data_loader.load_csv(dataset_path)

        # 移除索引列
        dataset = self.preprocessor.remove_index_column(dataset)

        # 清洗数据
        dataset_cleaned, original_count, cleaned_count, removed_count = (
            self.preprocessor.clean_data(dataset)
        )

        # 分离特征和标签
        X_df, y_series, X, y = self.data_loader.split_features_labels(dataset_cleaned)

        # 分割数据集
        X_train, X_test, y_train, y_test = self.preprocessor.split_train_test(
            X, y, test_size=test_size, random_state=random_state
        )

        # 保存训练集和验证集到CSV文件
        feature_columns = X_df.columns
        label_column = y_series.name

        # 创建DataFrame以便保存
        train_df = pd.DataFrame(X_train, columns=feature_columns)
        train_df[label_column] = y_train

        validation_df = pd.DataFrame(X_test, columns=feature_columns)
        validation_df[label_column] = y_test

        train_csv_path = os.path.join(self.processed_dir, "train.csv")
        validation_csv_path = os.path.join(self.processed_dir, "validation.csv")

        self.data_loader.save_csv(train_df, train_csv_path)
        self.data_loader.save_csv(validation_df, validation_csv_path)

        # print(f"已保存训练集到: {train_csv_path} ({len(train_df)} 行)")
        # print(f"已保存验证集到: {validation_csv_path} ({len(validation_df)} 行)")

        # 特征缩放
        X_train_scaled, X_test_scaled = self.preprocessor.scale_features(
            X_train, X_test
        )

        # 标签编码（XGBoost需要）
        y_train_encoded, use_encoder = self.preprocessor.encode_labels(
            y_train, model_type
        )

        # 创建并训练模型
        logger.info(f"开始训练模型: {model_type}")
        classifier = self._get_model_classifier(model_type, **model_kwargs)
        classifier.fit(X_train_scaled, y_train_encoded)  # type: ignore
        logger.info(f"模型训练完成: {model_type}")

        # 保存模型
        model_filename = f"{model_type}_{datetime.now().strftime(TIMESTAMP_FORMAT)}{MODEL_FILE_EXTENSION}"
        model_path = os.path.join(self.models_dir, model_filename)

        logger.debug(f"保存模型到: {model_path}")
        # 保存模型和scaler
        model_data = {
            "classifier": classifier,
            "scaler": self.preprocessor.scaler,
            "label_encoder": self.preprocessor.label_encoder if use_encoder else None,
            "model_type": model_type,
        }
        try:
            joblib.dump(model_data, model_path)
            logger.info(f"模型保存成功: {model_path}")
        except Exception as e:
            logger.error(f"模型保存失败: {model_path}, 错误: {str(e)}")
            raise

        features_count = X.shape[1]

        return {
            "classifier": classifier,
            "preprocessor": self.preprocessor,
            "model_path": model_path,
            "model_type": model_type,
            "features_count": features_count,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "X_train_scaled": X_train_scaled,
            "X_test_scaled": X_test_scaled,
            "y_train_encoded": y_train_encoded,
            "use_encoder": use_encoder,
            "data_cleaning": {
                "original_samples": original_count,
                "cleaned_samples": cleaned_count,
                "removed_samples": removed_count,
            },
        }
