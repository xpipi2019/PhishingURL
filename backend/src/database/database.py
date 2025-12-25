"""
数据库操作模块

提供数据库操作的封装类，包括：
- 模型版本管理（创建、查询、激活、删除）
- 预测历史管理（创建、查询、统计）
- 数据库连接管理（会话管理、连接检查）

使用SQLAlchemy ORM进行数据库操作，支持SQLite和PostgreSQL。
"""

from contextlib import contextmanager
from typing import Generator, List, Optional

from sqlalchemy.orm import Session

from ..utils import logger
from .models import (
    ModelVersion,
    PredictionHistory,
    SessionLocal,
    engine,
    init_db,
)


class Database:
    """
    数据库操作类
    
    提供统一的数据库操作接口，包括：
    - 模型版本管理
    - 预测历史管理
    - 数据库连接管理
    
    使用上下文管理器确保数据库会话的正确关闭和事务提交。
    """

    def __init__(self):
        """
        初始化数据库
        
        创建数据库引擎和会话工厂，并初始化数据库表结构。
        如果表已存在，则不会重复创建。
        """
        self.engine = engine
        self.SessionLocal = SessionLocal
        # 初始化数据库表（如果不存在则创建）
        init_db()
        logger.info("数据库初始化完成")

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """获取数据库会话上下文管理器"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"数据库操作失败: {str(e)}")
            raise
        finally:
            session.close()

    # ========== 模型版本管理 ==========

    def create_model_version(
        self,
        model_type: str,
        version: str,
        model_path: str,
        image_path: Optional[str],
        additional_images: Optional[dict],
        accuracy: float,
        confusion_matrix: Optional[list],
        classification_report: Optional[dict],
        training_time: float,
        train_samples: int,
        test_samples: int,
        features_count: int,
        data_cleaning: Optional[dict],
        is_best: bool = False,
    ) -> ModelVersion:
        """创建模型版本"""
        with self.get_session() as session:
            # 如果这是最佳模型，将其他模型标记为非最佳
            if is_best:
                session.query(ModelVersion).filter(
                    ModelVersion.model_type == model_type
                ).update({"is_best": False})

            # 将同类型的其他模型标记为非激活
            session.query(ModelVersion).filter(
                ModelVersion.model_type == model_type
            ).update({"is_active": False})

            model_version = ModelVersion(
                model_type=model_type,
                version=version,
                model_path=model_path,
                image_path=image_path,
                additional_images=additional_images,
                accuracy=accuracy,
                confusion_matrix=confusion_matrix,
                classification_report=classification_report,
                training_time=training_time,
                train_samples=train_samples,
                test_samples=test_samples,
                features_count=features_count,
                data_cleaning=data_cleaning,
                is_active=True,
                is_best=is_best,
            )
            session.add(model_version)
            session.flush()
            session.refresh(model_version)
            return model_version

    def get_model_version(
        self, model_type: str, version: Optional[str] = None
    ) -> Optional[ModelVersion]:
        """获取模型版本"""
        with self.get_session() as session:
            if version:
                return (
                    session.query(ModelVersion)
                    .filter(
                        ModelVersion.model_type == model_type,
                        ModelVersion.version == version,
                    )
                    .first()
                )
            else:
                # 返回最新的激活版本
                return (
                    session.query(ModelVersion)
                    .filter(
                        ModelVersion.model_type == model_type,
                        ModelVersion.is_active.is_(True),
                    )
                    .order_by(ModelVersion.created_at.desc())
                    .first()
                )

    def get_best_model_version(self) -> Optional[ModelVersion]:
        """获取最佳模型版本"""
        with self.get_session() as session:
            return (
                session.query(ModelVersion)
                .filter(ModelVersion.is_best.is_(True))
                .first()
            )

    def list_model_versions(
        self, model_type: Optional[str] = None, limit: int = 100
    ) -> List[ModelVersion]:
        """列出模型版本"""
        with self.get_session() as session:
            query = session.query(ModelVersion)
            if model_type:
                query = query.filter(ModelVersion.model_type == model_type)
            return query.order_by(ModelVersion.created_at.desc()).limit(limit).all()

    def activate_model_version(self, model_type: str, version: str) -> bool:
        """激活指定版本的模型"""
        with self.get_session() as session:
            # 将同类型的其他模型标记为非激活
            session.query(ModelVersion).filter(
                ModelVersion.model_type == model_type
            ).update({"is_active": False})

            # 激活指定版本
            result = (
                session.query(ModelVersion)
                .filter(
                    ModelVersion.model_type == model_type,
                    ModelVersion.version == version,
                )
                .update({"is_active": True})
            )
            return result > 0

    def delete_model_version(self, model_type: str, version: str) -> bool:
        """删除模型版本"""
        with self.get_session() as session:
            result = (
                session.query(ModelVersion)
                .filter(
                    ModelVersion.model_type == model_type,
                    ModelVersion.version == version,
                )
                .delete()
            )
            return result > 0

    # ========== 预测历史管理 ==========

    def create_prediction_history(
        self,
        url: str,
        model_type: str,
        model_version: Optional[str],
        prediction: str,
        probabilities: Optional[dict],
        is_safe: bool,
        features_used: Optional[list] = None,
        response_time_ms: Optional[float] = None,
    ) -> PredictionHistory:
        """创建预测历史记录"""
        with self.get_session() as session:
            prediction_history = PredictionHistory(
                url=url,
                model_type=model_type,
                model_version=model_version,
                prediction=prediction,
                probabilities=probabilities,
                is_safe=is_safe,
                features_used=features_used,
                response_time_ms=response_time_ms,
            )
            session.add(prediction_history)
            session.flush()
            session.refresh(prediction_history)
            return prediction_history

    def get_prediction_history(
        self, limit: int = 100, offset: int = 0, url: Optional[str] = None
    ) -> List[PredictionHistory]:
        """获取预测历史"""
        with self.get_session() as session:
            query = session.query(PredictionHistory)
            if url:
                query = query.filter(PredictionHistory.url == url)
            return (
                query.order_by(PredictionHistory.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )

    def get_prediction_stats(self) -> dict:
        """获取预测统计信息"""
        with self.get_session() as session:
            total = session.query(PredictionHistory).count()
            safe_count = (
                session.query(PredictionHistory)
                .filter(PredictionHistory.is_safe.is_(True))
                .count()
            )
            unsafe_count = total - safe_count

            return {
                "total_predictions": total,
                "safe_predictions": safe_count,
                "unsafe_predictions": unsafe_count,
                "safe_ratio": safe_count / total if total > 0 else 0,
            }

    def check_connection(self) -> bool:
        """检查数据库连接"""
        try:
            with self.get_session() as session:
                from sqlalchemy import text

                session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"数据库连接检查失败: {str(e)}")
            return False


# 创建全局数据库实例
db = Database()


def get_db() -> Database:
    """获取数据库实例"""
    return db
