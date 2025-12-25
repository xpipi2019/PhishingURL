"""
项目常量定义
提取所有魔法数字和硬编码值
"""

from typing import Dict, List

# ========== 模型训练相关常量 ==========

# 默认测试集比例
DEFAULT_TEST_SIZE: float = 0.25

# 默认随机种子
DEFAULT_RANDOM_STATE: int = 0

# 测试集比例范围
TEST_SIZE_MIN: float = 0.1
TEST_SIZE_MAX: float = 0.5

# KNN默认参数
KNN_DEFAULT_NEIGHBORS: int = 5
KNN_DEFAULT_METRIC: str = "minkowski"
KNN_DEFAULT_P: int = 2

# 随机森林默认参数
RANDOM_FOREST_DEFAULT_ESTIMATORS: int = 10

# ========== 数据预处理相关常量 ==========

# 有效特征值范围
VALID_FEATURE_VALUES: List[int] = [-1, 0, 1]

# ========== 特征提取相关常量 ==========

# URL长度阈值（字符数）
URL_LENGTH_SHORT_THRESHOLD: int = 54
URL_LENGTH_LONG_THRESHOLD: int = 75

# 默认特征数量
DEFAULT_FEATURES_COUNT: int = 30

# HTTP请求默认超时时间（秒）
DEFAULT_HTTP_TIMEOUT: int = 10

# User-Agent字符串
DEFAULT_USER_AGENT: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# ========== 预测相关常量 ==========

# 安全URL的预测值标识
SAFE_PREDICTION_VALUES: List[str] = ["1", "legitimate", "safe", "benign", "good"]

# ========== 健康检查相关常量 ==========

# CPU使用率告警阈值（百分比）
CPU_WARNING_THRESHOLD: float = 80.0

# ========== 模型评估相关常量 ==========

# 混淆矩阵颜色映射
CMAP_COLORS: str = "Blues"

# 图表DPI
FIGURE_DPI: int = 100

# 图表尺寸（英寸）
FIGURE_SIZE: tuple = (10, 8)

# ========== 文件路径相关常量 ==========

# 模型文件扩展名
MODEL_FILE_EXTENSION: str = ".joblib"

# 图片文件扩展名
IMAGE_FILE_EXTENSION: str = ".png"

# CSV文件扩展名
CSV_FILE_EXTENSION: str = ".csv"

# ========== 时间格式常量 ==========

# 时间戳格式
TIMESTAMP_FORMAT: str = "%Y%m%d_%H%M%S"

# 日期格式
DATE_FORMAT: str = "%Y-%m-%d"

# ISO时间格式
ISO_DATETIME_FORMAT: str = "%Y-%m-%dT%H:%M:%S"

# ========== 模型类型常量 ==========

# 支持的模型类型列表
SUPPORTED_MODEL_TYPES: List[str] = [
    "logistic_regression",
    "knn",
    "svm",
    "kernel_svm",
    "naive_bayes",
    "decision_tree",
    "random_forest",
    "xgboost",
]

# 模型类型显示名称映射
MODEL_TYPE_DISPLAY_NAMES: Dict[str, str] = {
    "logistic_regression": "逻辑回归",
    "knn": "K近邻",
    "svm": "支持向量机（线性）",
    "kernel_svm": "支持向量机（核）",
    "naive_bayes": "朴素贝叶斯",
    "decision_tree": "决策树",
    "random_forest": "随机森林",
    "xgboost": "XGBoost",
}

# ========== 数据验证相关常量 ==========

# 数据清洗：有效特征值范围
VALID_FEATURE_RANGE: tuple = (-1, 1)  # 包含 -1, 0, 1

# ========== 日志相关常量 ==========

# 日志级别
LOG_LEVELS: List[str] = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# 日志格式选项
LOG_FORMATS: List[str] = ["json", "text"]

# ========== API相关常量 ==========

# HTTP状态码
HTTP_OK: int = 200
HTTP_BAD_REQUEST: int = 400
HTTP_NOT_FOUND: int = 404
HTTP_INTERNAL_SERVER_ERROR: int = 500

# API响应消息
MESSAGE_TRAIN_SUCCESS: str = "模型训练完成"
MESSAGE_TRAIN_BEST_MODEL: str = "这是当前最佳模型"
MESSAGE_PREDICT_SUCCESS: str = "预测完成"
MESSAGE_MODEL_NOT_FOUND: str = "模型不存在"
MESSAGE_NO_MODEL_AVAILABLE: str = "没有可用的模型，请先训练模型"

# ========== 数据库相关常量 ==========

# 数据库查询默认限制
DEFAULT_QUERY_LIMIT: int = 100

# 模型版本列表默认限制
DEFAULT_VERSION_LIMIT: int = 50

# ========== 批量处理相关常量 ==========

# 批量预测默认最大数量
DEFAULT_BATCH_MAX_SIZE: int = 100

# 批量预测默认超时时间（秒）
DEFAULT_BATCH_TIMEOUT: int = 300

