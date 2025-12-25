"""
数据预处理模块

提供数据清洗、特征缩放、标签编码等功能。
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..constants import DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE, VALID_FEATURE_VALUES
from ..utils import logger


class DataPreprocessor:
    """
    数据预处理器

    负责数据清洗、特征缩放、标签编码等预处理操作。

    Attributes:
        scaler: StandardScaler实例，用于特征标准化
        label_encoder: LabelEncoder实例，用于标签编码（XGBoost需要）
    """

    def __init__(self) -> None:
        """
        初始化数据预处理器

        创建空的scaler和label_encoder，在训练时才会初始化。
        """
        self.scaler: StandardScaler | None = None
        self.label_encoder: LabelEncoder | None = None

    def remove_index_column(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        移除索引列Index（不区分大小写，去除空格后比较）

        某些数据集可能包含名为"Index"的列，该列不是特征，需要移除。

        Args:
            dataset: 原始数据集DataFrame

        Returns:
            pd.DataFrame: 移除索引列后的数据集

        Note:
            会查找所有列名（去除空格并转小写）为"index"的列并移除
        """
        index_cols = [col for col in dataset.columns if col.strip().lower() == "index"]
        if index_cols:
            logger.info(f"移除索引列: {index_cols}")
            dataset = dataset.drop(columns=index_cols)
        else:
            logger.debug("未发现索引列")
        return dataset

    def clean_data(self, dataset: pd.DataFrame) -> Tuple[pd.DataFrame, int, int, int]:
        """
        清洗数据：保留所有特征列的值都在有效范围内的行，且没有空值

        有效特征值范围：-1, 0, 1
        最后一列被视为标签列，不参与清洗验证。

        Args:
            dataset: 原始数据集DataFrame，最后一列应为标签

        Returns:
            Tuple[pd.DataFrame, int, int, int]:
                - 清洗后的数据集
                - 原始行数
                - 清洗后行数
                - 移除行数

        Raises:
            ValueError: 当清洗后没有有效数据时抛出异常

        Examples:
            >>> preprocessor = DataPreprocessor()
            >>> cleaned_df, original, cleaned, removed = preprocessor.clean_data(df)
            >>> print(f"移除了 {removed} 行无效数据")
        """
        # 分离特征和标签（最后一列为标签）
        feature_columns = dataset.columns[:-1]

        # 初始化有效掩码
        valid_mask = pd.Series([True] * len(dataset), index=dataset.index)

        # 检查空值
        for col in feature_columns:
            valid_mask = valid_mask & dataset[col].notna()

        # 检查特征值是否在有效范围内（-1, 0, 1）
        for col in feature_columns:
            valid_mask = valid_mask & dataset[col].isin(VALID_FEATURE_VALUES)

        # 应用过滤条件
        original_count = len(dataset)
        dataset_cleaned = dataset[valid_mask].copy()
        cleaned_count = len(dataset_cleaned)
        removed_count = original_count - cleaned_count

        if removed_count > 0:
            logger.warning(
                f"数据清洗完成: 原始数据 {original_count} 行，清洗后 {cleaned_count} 行，移除 {removed_count} 行无效数据"
            )
        else:
            logger.info(f"数据清洗完成: 所有 {original_count} 行数据均有效")

        if cleaned_count == 0:
            logger.error("数据清洗后没有有效数据，无法进行训练")
            raise ValueError("数据清洗后没有有效数据，无法进行训练")

        # 确保返回 DataFrame 类型
        if not isinstance(dataset_cleaned, pd.DataFrame):
            dataset_cleaned = pd.DataFrame(dataset_cleaned)
        return dataset_cleaned, original_count, cleaned_count, removed_count

    def split_train_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = DEFAULT_TEST_SIZE,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        分割训练集和测试集

        使用sklearn的train_test_split进行数据分割，保证结果可复现。

        Args:
            X: 特征数组，形状为 (n_samples, n_features)
            y: 标签数组，形状为 (n_samples,)
            test_size: 测试集比例，范围 [0.0, 1.0]，默认 0.25
            random_state: 随机种子，用于保证结果可复现，默认 0

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                (X_train, X_test, y_train, y_test)
                - X_train: 训练集特征
                - X_test: 测试集特征
                - y_train: 训练集标签
                - y_test: 测试集标签

        Examples:
            >>> X_train, X_test, y_train, y_test = preprocessor.split_train_test(
            ...     X, y, test_size=0.2, random_state=42
            ... )
        """
        logger.debug(f"分割数据集: test_size={test_size}, random_state={random_state}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        # 确保返回 np.ndarray 类型
        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
        logger.info(
            f"数据集分割完成: 训练集 {len(X_train)} 样本, 测试集 {len(X_test)} 样本"
        )
        return X_train, X_test, y_train, y_test

    def scale_features(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        特征标准化（Z-score标准化）

        使用StandardScaler对特征进行标准化：
        - 在训练集上fit，学习均值和标准差
        - 对训练集和测试集都进行transform
        - 公式: (x - mean) / std

        Args:
            X_train: 训练集特征数组，形状为 (n_train_samples, n_features)
            X_test: 测试集特征数组，形状为 (n_test_samples, n_features)

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                (X_train_scaled, X_test_scaled)
                - X_train_scaled: 标准化后的训练集特征
                - X_test_scaled: 标准化后的测试集特征

        Note:
            会更新 self.scaler 属性，保存标准化器以便后续使用
        """
        logger.debug("开始特征缩放")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        # 确保返回 np.ndarray 类型
        X_train_scaled = np.asarray(X_train_scaled)
        X_test_scaled = np.asarray(X_test_scaled)
        logger.info("特征缩放完成")
        return X_train_scaled, X_test_scaled

    def encode_labels(
        self, y_train: np.ndarray, model_type: str
    ) -> Tuple[np.ndarray, bool]:
        """
        标签编码

        某些模型（如XGBoost）需要数值型标签，需要将字符串标签编码为整数。

        Args:
            y_train: 训练集标签数组，形状为 (n_samples,)
            model_type: 模型类型，只有"xgboost"需要编码

        Returns:
            Tuple[np.ndarray, bool]:
                - 编码后的标签数组（如果是xgboost）或原始标签数组
                - 是否使用了编码器（True表示使用了，False表示未使用）

        Note:
            会更新 self.label_encoder 属性（如果使用了编码器）
        """
        if model_type == "xgboost":
            logger.debug(f"为模型 {model_type} 进行标签编码")
            self.label_encoder = LabelEncoder()
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            # 确保返回 np.ndarray 类型
            y_train_encoded = np.asarray(y_train_encoded)
            if self.label_encoder.classes_ is not None:
                logger.info(f"标签编码完成，类别数: {len(self.label_encoder.classes_)}")
            else:
                logger.info("标签编码完成")
            return y_train_encoded, True
        else:
            logger.debug(f"模型 {model_type} 不需要标签编码")
            # 确保返回 np.ndarray 类型
            y_train = np.asarray(y_train)
            return y_train, False

    def decode_labels(self, y_pred: np.ndarray, model_type: str) -> np.ndarray:
        """
        解码标签

        将编码后的数值标签转换回原始字符串标签。
        只有XGBoost模型需要解码，其他模型直接返回原标签。

        Args:
            y_pred: 预测标签数组，可能是编码后的数值或原始标签
            model_type: 模型类型，用于判断是否需要解码

        Returns:
            np.ndarray: 解码后的标签数组（如果是xgboost）或原始标签数组

        Note:
            需要先调用 encode_labels 创建 label_encoder
        """
        if model_type == "xgboost" and self.label_encoder is not None:
            logger.debug(f"为模型 {model_type} 解码标签")
            return self.label_encoder.inverse_transform(y_pred)
        else:
            return y_pred
