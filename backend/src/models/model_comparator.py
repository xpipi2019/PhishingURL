"""
模型对比模块
"""

import os
from typing import Dict, Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")  # 使用非交互式后端
import matplotlib.pyplot as plt

from ..utils import logger


class ModelComparator:
    """模型对比器"""

    def __init__(self, images_dir: str = "images"):
        """
        初始化模型对比器

        Args:
            images_dir: 图片保存目录
        """
        self.images_dir = images_dir
        os.makedirs(images_dir, exist_ok=True)
        logger.debug(f"初始化模型对比器: images_dir={images_dir}")

    def generate_comparison_plots(
        self, model_info: Dict[str, Dict], best_model_name: Optional[str] = None
    ) -> Dict[str, str]:
        """
        生成所有模型的对比图片

        Args:
            model_info: 模型信息字典
            best_model_name: 最佳模型名称

        Returns:
            包含对比图片路径的字典
        """
        if len(model_info) < 2:
            logger.warning(f"模型数量不足，无法生成对比图: 当前 {len(model_info)} 个模型，至少需要2个")
            return {}  # 至少需要2个模型才能对比
        
        logger.info(f"开始生成模型对比图: 共 {len(model_info)} 个模型")

        comparison_images = {}

        # 准备数据
        model_types = []
        accuracies = []
        training_times = []
        precisions = []
        recalls = []
        f1_scores = []

        for model_type, info in model_info.items():
            model_types.append(model_type)
            accuracies.append(info["accuracy"])
            training_times.append(info["training_time"])

            # 从分类报告中提取指标
            report = info["classification_report"]
            if "weighted avg" in report:
                precisions.append(report["weighted avg"]["precision"])
                recalls.append(report["weighted avg"]["recall"])
                f1_scores.append(report["weighted avg"]["f1-score"])
            else:
                precisions.append(info["accuracy"])
                recalls.append(info["accuracy"])
                f1_scores.append(info["accuracy"])

        # 1. 准确率对比图
        try:
            plt.figure(figsize=(12, 6))
            colors = [
                "gold" if model_type == best_model_name else "steelblue"
                for model_type in model_types
            ]
            bars = plt.bar(
                model_types, accuracies, color=colors, alpha=0.7, edgecolor="black"
            )
            plt.xlabel("Model Type", fontsize=12)
            plt.ylabel("Accuracy", fontsize=12)
            plt.title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
            plt.ylim([0, 1])
            plt.xticks(rotation=45, ha="right")
            plt.grid(axis="y", alpha=0.3)

            # 添加数值标签
            for i, (bar, acc) in enumerate(zip(bars, accuracies)):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{acc:.3f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            # 标注最佳模型
            if best_model_name:
                best_idx = model_types.index(best_model_name)
                plt.text(
                    best_idx,
                    accuracies[best_idx] + 0.05,
                    "★ Best Model",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color="gold",
                )

            plt.tight_layout()
            acc_comp_path = os.path.join(self.images_dir, "comparison_accuracy.png")
            plt.savefig(acc_comp_path, dpi=300, bbox_inches="tight")
            plt.close()
            comparison_images["accuracy_comparison"] = acc_comp_path
            logger.debug(f"准确率对比图保存成功: {acc_comp_path}")
        except Exception as e:
            logger.error(f"生成准确率对比图失败: {e}")

        # 2. 训练时间对比图
        try:
            plt.figure(figsize=(12, 6))
            colors = [
                "gold" if model_type == best_model_name else "steelblue"
                for model_type in model_types
            ]
            bars = plt.bar(
                model_types, training_times, color=colors, alpha=0.7, edgecolor="black"
            )
            plt.xlabel("Model Type", fontsize=12)
            plt.ylabel("Training Time (seconds)", fontsize=12)
            plt.title("Model Training Time Comparison", fontsize=14, fontweight="bold")
            plt.xticks(rotation=45, ha="right")
            plt.grid(axis="y", alpha=0.3)

            # 添加数值标签
            for bar, time_val in zip(bars, training_times):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(training_times) * 0.01,
                    f"{time_val:.2f}s",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            plt.tight_layout()
            time_comp_path = os.path.join(
                self.images_dir, "comparison_training_time.png"
            )
            plt.savefig(time_comp_path, dpi=300, bbox_inches="tight")
            plt.close()
            comparison_images["training_time_comparison"] = time_comp_path
            logger.debug(f"训练时间对比图保存成功: {time_comp_path}")
        except Exception as e:
            logger.error(f"生成训练时间对比图失败: {e}")

        # 3. 综合指标雷达图
        try:
            # 归一化指标到0-1范围
            max_acc = max(accuracies)
            max_prec = max(precisions)
            max_rec = max(recalls)
            max_f1 = max(f1_scores)

            # 创建雷达图
            categories = ["Accuracy", "Precision", "Recall", "F1-Score"]
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]  # 闭合

            fig, ax = plt.subplots(
                figsize=(10, 10), subplot_kw=dict(projection="polar")
            )

            # 为每个模型绘制雷达图
            import matplotlib.cm as cm

            try:
                # 尝试使用get_cmap（新版本matplotlib）
                colors_list = cm.get_cmap("Set3")(np.linspace(0, 1, len(model_types)))
            except (AttributeError, ValueError):
                # 如果get_cmap不存在或失败，使用tab10 colormap
                colors_list = cm.get_cmap("tab10")(np.linspace(0, 1, len(model_types)))
            for idx, model_type in enumerate(model_types):
                values = [
                    accuracies[idx] / max_acc,
                    precisions[idx] / max_prec,
                    recalls[idx] / max_rec,
                    f1_scores[idx] / max_f1,
                ]
                values += values[:1]  # 闭合

                # 如果是最佳模型，使用特殊样式
                if model_type == best_model_name:
                    ax.plot(
                        angles,
                        values,
                        "o-",
                        linewidth=3,
                        label=model_type,
                        color="gold",
                        alpha=0.8,
                    )
                    ax.fill(angles, values, alpha=0.25, color="gold")
                else:
                    ax.plot(
                        angles,
                        values,
                        "o-",
                        linewidth=2,
                        label=model_type,
                        color=colors_list[idx],
                        alpha=0.7,
                    )
                    ax.fill(angles, values, alpha=0.15, color=colors_list[idx])

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1)
            ax.set_title(
                "Model Performance Comparison (Radar Chart)",
                size=14,
                fontweight="bold",
                pad=20,
            )
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
            ax.grid(True)

            plt.tight_layout()
            radar_path = os.path.join(self.images_dir, "comparison_radar.png")
            plt.savefig(radar_path, dpi=300, bbox_inches="tight")
            plt.close()
            comparison_images["radar_comparison"] = radar_path
            logger.debug(f"雷达图保存成功: {radar_path}")
        except Exception as e:
            logger.error(f"生成雷达图失败: {e}")

        # 4. 多指标对比图
        try:
            x = np.arange(len(model_types))
            width = 0.25

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.bar(x - width, accuracies, width, label="Accuracy", alpha=0.8)
            ax.bar(x, precisions, width, label="Precision", alpha=0.8)
            ax.bar(x + width, recalls, width, label="Recall", alpha=0.8)

            ax.set_xlabel("Model Type", fontsize=12)
            ax.set_ylabel("Score", fontsize=12)
            ax.set_title(
                "Model Performance Metrics Comparison", fontsize=14, fontweight="bold"
            )
            ax.set_xticks(x)
            ax.set_xticklabels(model_types, rotation=45, ha="right")
            ax.legend()
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.3)

            # 标注最佳模型
            if best_model_name:
                best_idx = model_types.index(best_model_name)
                ax.text(
                    best_idx,
                    1.05,
                    "★ Best",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color="gold",
                )

            plt.tight_layout()
            metrics_path = os.path.join(self.images_dir, "comparison_metrics.png")
            plt.savefig(metrics_path, dpi=300, bbox_inches="tight")
            plt.close()
            comparison_images["metrics_comparison"] = metrics_path
            logger.debug(f"多指标对比图保存成功: {metrics_path}")
        except Exception as e:
            logger.error(f"生成多指标对比图失败: {e}")

        return comparison_images

