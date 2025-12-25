"""
API接口模块
包括路由定义、输入验证、响应格式化
"""

from .routes import router
from .schemas import (
    AllModelsResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelStatusResponse,
    ModelVersionInfo,
    ModelVersionsResponse,
    PredictRequest,
    PredictResponse,
    PredictionHistoryItem,
    PredictionHistoryResponse,
    PredictionStatsResponse,
    TrainRequest,
    TrainResponse,
)

__all__ = [
    "router",
    "TrainRequest",
    "TrainResponse",
    "PredictRequest",
    "PredictResponse",
    "BatchPredictRequest",
    "BatchPredictResponse",
    "ModelStatusResponse",
    "HealthResponse",
    "AllModelsResponse",
    "ModelVersionInfo",
    "ModelVersionsResponse",
    "PredictionHistoryItem",
    "PredictionHistoryResponse",
    "PredictionStatsResponse",
]

