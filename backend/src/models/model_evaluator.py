"""
模型评估模块
"""

import os
from typing import Any, Dict, Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from ..utils import logger


class ModelEvaluator:
    """模型评估器"""

    def __init__(self, images_dir: str = "images"):
        """
        初始化模型评估器

        Args:
            images_dir: 图片保存目录
        """
        self.images_dir = images_dir
        os.makedirs(images_dir, exist_ok=True)
        logger.debug(f"初始化模型评估器: images_dir={images_dir}")

    def evaluate(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        model_type: str,
        timestamp: str,
    ) -> Dict[str, Any]:
        """
        评估模型

        Args:
            y_test: 真实标签
            y_pred: 预测标签
            model_type: 模型类型
            timestamp: 时间戳

        Returns:
            评估结果字典
        """
        logger.info(f"开始评估模型: {model_type}")
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        logger.info(f"模型评估完成: {model_type}, 准确率: {accuracy:.4f}")

        # 生成混淆矩阵图片
        image_filename = f"{model_type}_{timestamp}.png"
        image_path = os.path.join(self.images_dir, image_filename)
        logger.debug(f"生成混淆矩阵图片: {image_path}")
        self._save_confusion_matrix(cm, y_test, image_path, model_type)

        return {
            "accuracy": float(accuracy),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "image_path": image_path,
        }

    def _save_confusion_matrix(
        self, cm: np.ndarray, y_test: np.ndarray, image_path: str, model_type: str
    ):
        """保存混淆矩阵热力图"""
        # 归一化混淆矩阵，避免除零错误
        row_sums = cm.sum(axis=1)
        # 处理除零情况：如果某行全为0，则保持原值
        data = cm.astype("float")
        for i in range(len(row_sums)):
            if row_sums[i] > 0:
                data[i] = data[i] / row_sums[i]

        unique_labels = np.unique(y_test)
        # 确保DataFrame的行列标签匹配
        if len(unique_labels) != cm.shape[0]:
            # 如果标签数量不匹配，使用索引作为标签
            unique_labels = np.arange(cm.shape[0])

        df_cm = pd.DataFrame(data, columns=unique_labels, index=unique_labels)
        df_cm.index.name = "Actual"
        df_cm.columns.name = "Predicted"

        plt.figure(figsize=(6, 3))
        sns.set(font_scale=1.4)
        sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 10}, fmt=".2f")
        plt.title(f"Confusion Matrix Heat Map - {model_type}\n")
        plt.tight_layout()
        plt.savefig(image_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.debug(f"混淆矩阵图片保存成功: {image_path}")

    def generate_additional_plots(
        self,
        classifier: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        model_type: str,
        timestamp: str,
        label_encoder: Optional[Any] = None,
    ) -> Dict[str, str]:
        """
        生成额外的评估图片

        Args:
            classifier: 训练好的分类器
            X_test: 测试集特征
            y_test: 测试集真实标签
            y_pred: 测试集预测标签
            model_type: 模型类型
            timestamp: 时间戳
            label_encoder: 标签编码器（用于XGBoost）

        Returns:
            包含额外图片路径的字典
        """
        additional_images = {}
        unique_labels = np.unique(y_test)

        # 1. 生成ROC曲线（如果是二分类）
        try:
            if len(unique_labels) == 2:
                # 获取预测概率
                if hasattr(classifier, "predict_proba"):
                    y_proba = classifier.predict_proba(X_test)
                    if label_encoder is not None:
                        # 找到正类标签的索引
                        pos_label_idx = np.where(
                            label_encoder.classes_ == unique_labels[1]
                        )[0][0]
                        y_scores = y_proba[:, pos_label_idx]
                    else:
                        # 假设第二个类别是正类
                        y_scores = (
                            y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
                        )

                    # 将标签转换为0和1
                    y_test_binary = (y_test == unique_labels[1]).astype(int)

                    # 计算ROC曲线
                    fpr, tpr, _ = roc_curve(y_test_binary, y_scores)
                    roc_auc = auc(fpr, tpr)

                    # 绘制ROC曲线
                    plt.figure(figsize=(8, 6))
                    plt.plot(
                        fpr,
                        tpr,
                        color="darkorange",
                        lw=2,
                        label=f"ROC curve (AUC = {roc_auc:.2f})",
                    )
                    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title(f"ROC Curve - {model_type}")
                    plt.legend(loc="lower right")
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    roc_path = os.path.join(
                        self.images_dir, f"{model_type}_{timestamp}_roc.png"
                    )
                    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
                    plt.close()
                    additional_images["roc_curve"] = roc_path

                    # 2. 生成Precision-Recall曲线
                    precision, recall, _ = precision_recall_curve(
                        y_test_binary, y_scores
                    )
                    pr_auc = auc(recall, precision)

                    plt.figure(figsize=(8, 6))
                    plt.plot(
                        recall,
                        precision,
                        color="blue",
                        lw=2,
                        label=f"PR curve (AUC = {pr_auc:.2f})",
                    )
                    plt.xlabel("Recall")
                    plt.ylabel("Precision")
                    plt.title(f"Precision-Recall Curve - {model_type}")
                    plt.legend(loc="lower left")
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    pr_path = os.path.join(
                        self.images_dir, f"{model_type}_{timestamp}_pr.png"
                    )
                    plt.savefig(pr_path, dpi=300, bbox_inches="tight")
                    plt.close()
                    additional_images["pr_curve"] = pr_path
                    logger.debug(f"PR曲线图片保存成功: {pr_path}")
        except Exception as e:
            logger.warning(f"生成ROC/PR曲线失败: {e}")

        # 3. 生成特征重要性图（如果模型支持）
        try:
            if hasattr(classifier, "feature_importances_"):
                importances = classifier.feature_importances_
                indices = np.argsort(importances)[::-1][:20]  # 只显示前20个重要特征

                plt.figure(figsize=(10, 6))
                plt.bar(range(len(indices)), importances[indices])
                plt.xlabel("Feature Index")
                plt.ylabel("Importance")
                plt.title(f"Top 20 Feature Importances - {model_type}")
                plt.xticks(range(len(indices)), [str(i) for i in indices], rotation=45)
                plt.tight_layout()
                fi_path = os.path.join(
                    self.images_dir, f"{model_type}_{timestamp}_feature_importance.png"
                )
                plt.savefig(fi_path, dpi=300, bbox_inches="tight")
                plt.close()
                additional_images["feature_importance"] = fi_path
                logger.debug(f"特征重要性图保存成功: {fi_path}")
        except Exception as e:
            logger.warning(f"生成特征重要性图失败: {e}")

        return additional_images

