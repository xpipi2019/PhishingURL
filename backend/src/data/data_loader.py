"""
数据加载模块

提供CSV文件加载、保存和特征标签分离功能。
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd

from ..utils import logger


class DataLoader:
    """
    数据加载器

    负责从CSV文件加载数据，保存处理后的数据，以及分离特征和标签。

    Note:
        所有方法都是静态方法，可以直接通过类调用。
    """

    @staticmethod
    def load_csv(file_path: str) -> pd.DataFrame:
        """
        从CSV文件加载数据

        使用pandas读取CSV文件，支持标准CSV格式。

        Args:
            file_path: CSV文件路径，可以是相对路径或绝对路径

        Returns:
            pd.DataFrame: 加载的数据DataFrame

        Raises:
            FileNotFoundError: 当文件不存在时
            Exception: 当文件格式错误或读取失败时

        Examples:
            >>> df = DataLoader.load_csv("data/raw/PhishingData.csv")
            >>> print(f"加载了 {len(df)} 行数据")
        """
        logger.debug(f"开始加载CSV文件: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"数据集文件不存在: {file_path}")
            raise FileNotFoundError(f"数据集文件不存在: {file_path}")

        try:
            df = pd.read_csv(file_path)
            logger.info(f"成功加载CSV文件: {file_path}, 数据形状: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"加载CSV文件失败: {file_path}, 错误: {str(e)}")
            raise

    @staticmethod
    def save_csv(dataframe: pd.DataFrame, file_path: str) -> None:
        """
        保存DataFrame到CSV文件

        将DataFrame保存为CSV格式，不包含索引列。

        Args:
            dataframe: 要保存的DataFrame
            file_path: 保存路径，如果目录不存在会自动创建

        Raises:
            Exception: 当保存失败时（如权限不足、磁盘空间不足等）

        Examples:
            >>> DataLoader.save_csv(df, "data/processed/train.csv")
        """
        logger.debug(f"开始保存CSV文件: {file_path}, 数据形状: {dataframe.shape}")
        try:
            dataframe.to_csv(file_path, index=False)
            logger.info(f"成功保存CSV文件: {file_path}, 数据行数: {len(dataframe)}")
        except Exception as e:
            logger.error(f"保存CSV文件失败: {file_path}, 错误: {str(e)}")
            raise

    @staticmethod
    def split_features_labels(
        dataset: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray]:
        """
        分离特征和标签

        将数据集分为特征和标签两部分：
        - 特征：除最后一列外的所有列
        - 标签：最后一列

        Args:
            dataset: 数据集DataFrame，最后一列必须是标签列

        Returns:
            Tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray]:
                - X_df: 特征DataFrame（保留列名）
                - y_series: 标签Series（保留列名）
                - X: 特征numpy数组（用于模型训练）
                - y: 标签numpy数组（用于模型训练）

        Examples:
            >>> X_df, y_series, X, y = DataLoader.split_features_labels(dataset)
            >>> print(f"特征数: {X.shape[1]}, 样本数: {X.shape[0]}")
        """
        logger.debug("开始分离特征和标签")
        feature_columns = dataset.columns[:-1]  # 除最后一列外的所有列
        label_column = dataset.columns[-1]  # 最后一列是标签

        X_df = dataset[feature_columns]
        # 确保返回 DataFrame 类型
        if not isinstance(X_df, pd.DataFrame):
            X_df = pd.DataFrame(X_df)
        y_series = dataset[label_column]
        # 确保返回 Series 类型
        if not isinstance(y_series, pd.Series):
            y_series = pd.Series(y_series)
        X = np.array(X_df)
        y = np.array(y_series)

        logger.info(
            f"特征数量: {len(feature_columns)}, 标签列: {label_column}, 样本数: {len(X)}"
        )
        return X_df, y_series, X, y
