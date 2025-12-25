"""
数据库模型定义
"""

from datetime import datetime
from typing import Any, Dict

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class ModelVersion(Base):
    """模型版本表"""

    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    model_type = Column(String(50), nullable=False, index=True)
    version = Column(String(50), nullable=False, index=True)
    model_path = Column(String(500), nullable=False)
    image_path = Column(String(500), nullable=True)
    additional_images = Column(JSON, nullable=True)
    accuracy = Column(Float, nullable=False)
    confusion_matrix = Column(JSON, nullable=True)
    classification_report = Column(JSON, nullable=True)
    training_time = Column(Float, nullable=False)
    train_samples = Column(Integer, nullable=False)
    test_samples = Column(Integer, nullable=False)
    features_count = Column(Integer, nullable=False)
    data_cleaning = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=False, index=True)
    is_best = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "model_type": self.model_type,
            "version": self.version,
            "model_path": self.model_path,
            "image_path": self.image_path,
            "additional_images": self.additional_images,
            "accuracy": self.accuracy,
            "confusion_matrix": self.confusion_matrix,
            "classification_report": self.classification_report,
            "training_time": self.training_time,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "features_count": self.features_count,
            "data_cleaning": self.data_cleaning,
            "is_active": self.is_active,
            "is_best": self.is_best,
            "created_at": self.created_at.isoformat()
            if self.created_at is not None
            else None,
            "updated_at": self.updated_at.isoformat()
            if self.updated_at is not None
            else None,
        }


class PredictionHistory(Base):
    """预测历史表"""

    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String(2000), nullable=False, index=True)
    model_type = Column(String(50), nullable=False, index=True)
    model_version = Column(String(50), nullable=True, index=True)
    prediction = Column(String(50), nullable=False)
    probabilities = Column(JSON, nullable=True)
    is_safe = Column(Boolean, nullable=False, index=True)
    features_used = Column(JSON, nullable=True)
    response_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "url": self.url,
            "model_type": self.model_type,
            "model_version": self.model_version,
            "prediction": self.prediction,
            "probabilities": self.probabilities,
            "is_safe": self.is_safe,
            "features_used": self.features_used,
            "response_time_ms": self.response_time_ms,
            "created_at": self.created_at.isoformat()
            if self.created_at is not None
            else None,
        }


# 创建数据库引擎和会话
def get_database_url() -> str:
    """获取数据库URL"""
    from ..config import settings

    db_url = settings.DATABASE_URL
    # 如果是SQLite，确保目录存在
    if db_url.startswith("sqlite"):
        # 提取路径
        if "///" in db_url:
            path = db_url.split("///")[1]
            from pathlib import Path

            Path(path).parent.mkdir(parents=True, exist_ok=True)
    return db_url


engine = create_engine(
    get_database_url(),
    connect_args={"check_same_thread": False} if "sqlite" in get_database_url() else {},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """初始化数据库，创建所有表"""
    Base.metadata.create_all(bind=engine)
