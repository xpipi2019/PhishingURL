"""
API数据模型定义
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator


class TrainRequest(BaseModel):
    """训练请求模型"""

    dataset_path: str = Field(..., description="数据集文件路径")
    model_type: str = Field(
        ...,
        description="模型类型",
        examples=[
            "logistic_regression",
            "knn",
            "svm",
            "kernel_svm",
            "naive_bayes",
            "decision_tree",
            "random_forest",
            "xgboost",
        ],
    )
    test_size: float = Field(0.25, ge=0.1, le=0.5, description="测试集比例")
    random_state: int = Field(0, description="随机种子")

    @field_validator("dataset_path")
    @classmethod
    def normalize_path(cls, v: str) -> str:
        """
        规范化路径，将反斜杠转换为正斜杠
        这样可以避免Windows路径在JSON中的转义问题
        """
        if isinstance(v, str):
            # 将反斜杠转换为正斜杠，并规范化路径
            normalized = v.replace("\\", "/")
            # 移除多余的正斜杠（保留开头的双斜杠，如 //server/path）
            if normalized.startswith("//"):
                parts = normalized[2:].split("/")
                normalized = "//" + "/".join(parts)
            else:
                parts = normalized.split("/")
                normalized = "/".join(parts)
            return normalized
        return v


class TrainResponse(BaseModel):
    """训练响应模型"""

    success: bool
    model_type: str
    model_path: str
    image_path: str
    accuracy: float
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]
    training_time: float
    is_best_model: bool
    message: Optional[str] = None


class PredictRequest(BaseModel):
    """预测请求模型"""

    url: HttpUrl = Field(..., description="要检测的URL")
    model_type: Optional[str] = Field(
        None, description="指定使用的模型类型，默认使用最佳模型"
    )


class PredictResponse(BaseModel):
    """预测响应模型"""

    url: str
    prediction: str
    probabilities: Dict[str, float]
    model_used: str
    is_safe: bool = Field(..., description="URL是否安全（根据预测结果推断）")


class BatchPredictRequest(BaseModel):
    """批量预测请求模型"""

    urls: List[HttpUrl] = Field(..., description="要检测的URL列表")
    model_type: Optional[str] = Field(
        None, description="指定使用的模型类型，默认使用最佳模型"
    )


class BatchPredictResponse(BaseModel):
    """批量预测响应模型"""

    results: List[Dict[str, Any]]
    total: int
    model_used: str


class ModelStatusResponse(BaseModel):
    """模型状态响应模型"""

    model_type: str
    model_path: str
    image_path: str
    additional_images: Dict[str, str] = Field(default_factory=dict)
    accuracy: float
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]
    training_time: float
    train_samples: int
    test_samples: int
    features_count: int
    created_at: str
    is_best_model: bool
    model_exists: bool
    image_exists: bool


class HealthResponse(BaseModel):
    """健康检查响应模型"""

    status: str
    has_trained_models: bool
    model_count: int
    best_model: Optional[str]
    best_accuracy: float
    can_predict: bool
    model_working: bool
    uptime_seconds: float
    uptime_formatted: str
    database_connected: Optional[bool] = None
    disk_healthy: Optional[bool] = None
    memory_healthy: Optional[bool] = None
    cpu_healthy: Optional[bool] = None
    system_health: Optional[Dict[str, Any]] = None


class ModelInfo(BaseModel):
    """单个模型信息"""

    model_type: str
    model_path: str
    image_path: str
    additional_images: Dict[str, str] = Field(default_factory=dict)
    accuracy: float
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]
    training_time: float
    train_samples: int
    test_samples: int
    features_count: int
    created_at: str
    is_best_model: bool
    model_exists: bool
    image_exists: bool


class AllModelsResponse(BaseModel):
    """所有模型信息响应模型"""

    total_models: int
    best_model: Optional[str]
    best_accuracy: float
    models: List[ModelInfo]
    comparison_images: Dict[str, str]
    updated_at: str


class ModelVersionInfo(BaseModel):
    """模型版本信息"""

    id: int
    model_type: str
    version: str
    model_path: str
    image_path: Optional[str]
    additional_images: Optional[Dict[str, Any]]
    accuracy: float
    confusion_matrix: Optional[List[List[int]]]
    classification_report: Optional[Dict[str, Any]]
    training_time: float
    train_samples: int
    test_samples: int
    features_count: int
    data_cleaning: Optional[Dict[str, Any]]
    is_active: bool
    is_best: bool
    created_at: str
    updated_at: Optional[str]


class ModelVersionsResponse(BaseModel):
    """模型版本列表响应"""

    model_type: str
    total_versions: int
    versions: List[ModelVersionInfo]
    active_version: Optional[str]
    best_version: Optional[str]


class PredictionHistoryItem(BaseModel):
    """预测历史项"""

    id: int
    url: str
    model_type: str
    model_version: Optional[str]
    prediction: str
    probabilities: Optional[Dict[str, float]]
    is_safe: bool
    response_time_ms: Optional[float]
    created_at: str


class PredictionHistoryResponse(BaseModel):
    """预测历史响应"""

    total: int
    limit: int
    offset: int
    history: List[PredictionHistoryItem]


class PredictionStatsResponse(BaseModel):
    """预测统计响应"""

    total_predictions: int
    safe_predictions: int
    unsafe_predictions: int
    safe_ratio: float
