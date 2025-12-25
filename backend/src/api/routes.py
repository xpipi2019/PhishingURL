"""
API路由定义
"""

import asyncio
import os
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from ..config import settings
from ..data import FeatureExtractor
from ..database import get_db
from ..database.models import PredictionHistory
from ..models import ModelManager, ModelPredictor, ModelTrainer
from ..utils import health_check, logger
from .schemas import (
    AllModelsResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelStatusResponse,
    ModelVersionInfo,
    ModelVersionsResponse,
    PredictionHistoryItem,
    PredictionHistoryResponse,
    PredictionStatsResponse,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
)

# 创建路由器
router = APIRouter(prefix="/api", tags=["api"])

# 初始化服务
logger.info("初始化API服务")
model_manager = ModelManager()
model_trainer = ModelTrainer()
feature_extractor = FeatureExtractor()
logger.info("API服务初始化完成")


@router.post("/train", response_model=TrainResponse, status_code=status.HTTP_200_OK)
async def train_model(request: TrainRequest):
    """
    训练模型

    - **dataset_path**: 数据集文件路径
    - **model_type**: 模型类型（logistic_regression, knn, svm, kernel_svm, naive_bayes, decision_tree, random_forest, xgboost）
    - **test_size**: 测试集比例（默认0.25）
    - **random_state**: 随机种子（默认0）

    训练完成后会：
    - 保存模型文件
    - 生成混淆矩阵图片
    - 记录模型信息
    - 更新最佳模型（如果准确率更高）
    """
    try:
        logger.info(
            f"收到训练请求: model_type={request.model_type}, dataset_path={request.dataset_path}"
        )
        # 验证模型类型
        if request.model_type not in ModelTrainer.MODEL_TYPES:
            logger.warning(f"不支持的模型类型: {request.model_type}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的模型类型: {request.model_type}。支持的模型类型: {list(ModelTrainer.MODEL_TYPES.keys())}",
            )

        # 训练模型
        import time
        from datetime import datetime

        from ..models import ModelEvaluator

        start_time = time.time()
        train_result = model_trainer.train(
            dataset_path=request.dataset_path,
            model_type=request.model_type,
            test_size=request.test_size,
            random_state=request.random_state,
        )

        # 解码预测标签
        y_pred_labels = train_result["preprocessor"].decode_labels(
            train_result["classifier"].predict(train_result["X_test_scaled"]),
            request.model_type,
        )

        # 评估模型
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        evaluator = ModelEvaluator()
        eval_result = evaluator.evaluate(
            train_result["y_test"],
            y_pred_labels,
            request.model_type,
            timestamp,
        )

        # 生成额外图片
        current_label_encoder = (
            train_result["preprocessor"].label_encoder
            if train_result["use_encoder"]
            else None
        )
        additional_images = evaluator.generate_additional_plots(
            train_result["classifier"],
            train_result["X_test_scaled"],
            train_result["y_test"],
            y_pred_labels,
            request.model_type,
            timestamp,
            current_label_encoder,
        )

        # 保存模型信息
        training_time = time.time() - start_time
        model_manager.save_model_info(
            model_type=request.model_type,
            model_path=train_result["model_path"],
            image_path=eval_result["image_path"],
            additional_images=additional_images,
            accuracy=eval_result["accuracy"],
            confusion_matrix=eval_result["confusion_matrix"],
            classification_report=eval_result["classification_report"],
            training_time=training_time,
            train_samples=len(train_result["X_train"]),
            test_samples=len(train_result["X_test"]),
            features_count=train_result["features_count"],
            data_cleaning=train_result["data_cleaning"],
        )

        result = {
            "success": True,
            "model_type": request.model_type,
            "model_path": train_result["model_path"],
            "image_path": eval_result["image_path"],
            "accuracy": eval_result["accuracy"],
            "confusion_matrix": eval_result["confusion_matrix"],
            "classification_report": eval_result["classification_report"],
            "training_time": training_time,
            "is_best_model": request.model_type == model_manager.best_model_name,
            "data_cleaning": train_result["data_cleaning"],
        }

        result["message"] = f"模型 {request.model_type} 训练完成"
        if result["is_best_model"]:
            result["message"] += "，这是当前最佳模型"

        logger.info(
            f"模型训练成功: {request.model_type}, 准确率: {result['accuracy']:.4f}, 训练时间: {result['training_time']:.2f}秒"
        )
        return TrainResponse(**result)

    except FileNotFoundError as e:
        logger.error(f"训练失败 - 文件不存在: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"数据集文件不存在: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"训练失败 - 未知错误: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"训练失败: {str(e)}",
        )


@router.post("/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK)
async def predict_url(request: PredictRequest):
    """
    预测URL是否可靠

    - **url**: 要检测的URL
    - **model_type**: 指定使用的模型类型（可选，默认使用最佳模型）

    后端会自动从URL提取特征向量，然后进行预测。
    返回URL的诊断信息，包括预测结果和概率分布。
    """
    start_time = time.time()
    try:
        logger.debug(
            f"收到预测请求: url={request.url}, model_type={request.model_type}"
        )
        # 获取模型信息
        model_info = model_manager.get_model_info(request.model_type)
        model_type_used = request.model_type or model_manager.best_model_name or ""
        model_version = model_info.get("version", "unknown")

        # 进行预测
        predictor = ModelPredictor(model_info)

        # 从URL自动提取特征
        logger.info(f"从URL自动提取特征: {request.url}")
        # 先提取特征，以便记录到数据库
        features = feature_extractor.extract_url_features(str(request.url))
        features_used = features.tolist()
        logger.debug(f"特征提取完成，特征数量: {len(features)}")

        # 使用提取的特征进行预测
        result = predictor.predict(features, model_type_used)

        # 判断是否安全（假设预测结果为'1'或'legitimate'等表示安全）
        prediction_str = str(result["prediction"]).lower()
        is_safe = prediction_str in ["1", "legitimate", "safe", "benign", "good"]

        response_time_ms = (time.time() - start_time) * 1000

        # 保存预测历史到数据库
        try:
            db = get_db()
            db.create_prediction_history(
                url=str(request.url),
                model_type=model_type_used,
                model_version=model_version,
                prediction=result["prediction"],
                probabilities=result["probabilities"],
                is_safe=is_safe,
                features_used=features_used,
                response_time_ms=response_time_ms,
            )
        except Exception as e:
            logger.warning(f"保存预测历史失败: {str(e)}")

        return PredictResponse(
            url=str(request.url),
            prediction=result["prediction"],
            probabilities=result["probabilities"],
            model_used=model_type_used,
            is_safe=is_safe,
        )

    except ValueError as e:
        logger.warning(f"预测失败 - 参数错误: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"预测失败 - 未知错误: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"预测失败: {str(e)}",
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    status_code=status.HTTP_200_OK,
)
async def predict_batch_urls(request: BatchPredictRequest):
    """
    批量预测URL是否可靠（异步处理）

    - **urls**: 要检测的URL列表（最多100个）
    - **model_type**: 指定使用的模型类型（可选，默认使用最佳模型）

    后端会自动从每个URL提取特征向量，然后进行预测。
    返回所有URL的诊断信息。
    """
    start_time = time.time()
    try:
        # 验证批量大小
        if len(request.urls) > settings.BATCH_PREDICT_MAX_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"批量预测数量超过限制: 最多{settings.BATCH_PREDICT_MAX_SIZE}个URL",
            )

        logger.debug(
            f"收到批量预测请求: urls数量={len(request.urls)}, model_type={request.model_type}"
        )

        # 获取模型信息
        model_info = model_manager.get_model_info(request.model_type)
        model_type_used = request.model_type or model_manager.best_model_name or ""
        model_version = model_info.get("version", "unknown")

        predictor = ModelPredictor(model_info)

        # 异步处理每个URL的预测
        async def predict_single_url(url):
            """单个URL的预测任务"""
            try:
                # 从URL自动提取特征
                logger.debug(f"从URL提取特征: {url}")
                features = feature_extractor.extract_url_features(str(url))
                features_used = features.tolist()
                logger.debug(f"特征提取完成，特征数量: {len(features)}")

                # 使用提取的特征进行预测
                result = predictor.predict(features, model_type_used)

                prediction_str = str(result["prediction"]).lower()
                is_safe = prediction_str in [
                    "1",
                    "legitimate",
                    "safe",
                    "benign",
                    "good",
                ]

                return {
                    "url": str(url),
                    "prediction": result["prediction"],
                    "probabilities": result["probabilities"],
                    "is_safe": is_safe,
                    "features_used": features_used,
                }
            except Exception as e:
                logger.error(f"URL预测失败: {url}, 错误: {str(e)}")
                return {
                    "url": str(url),
                    "prediction": "error",
                    "probabilities": {},
                    "is_safe": False,
                    "error": str(e),
                }

        # 为每个URL创建预测任务
        tasks = [predict_single_url(url) for url in request.urls]

        # 并发执行所有预测任务
        results = await asyncio.gather(*tasks)

        response_time_ms = (time.time() - start_time) * 1000

        # 保存预测历史到数据库（异步批量保存）
        try:
            db = get_db()
            for result_item in results:
                if "error" not in result_item:
                    db.create_prediction_history(
                        url=result_item["url"],
                        model_type=model_type_used,
                        model_version=model_version,
                        prediction=result_item["prediction"],
                        probabilities=result_item["probabilities"],
                        is_safe=result_item["is_safe"],
                        features_used=result_item.get("features_used"),
                        response_time_ms=response_time_ms
                        / len(results),  # 平均响应时间
                    )
        except Exception as e:
            logger.warning(f"保存批量预测历史失败: {str(e)}")

        return BatchPredictResponse(
            results=results,
            total=len(results),
            model_used=model_type_used,
        )

    except ValueError as e:
        logger.warning(f"批量预测失败 - 参数错误: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"批量预测失败 - 未知错误: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"批量预测失败: {str(e)}",
        )


@router.get(
    "/model/{model_type}/status",
    response_model=ModelStatusResponse,
    status_code=status.HTTP_200_OK,
)
async def get_model_status(model_type: str):
    """
    获取指定模型的状态信息

    - **model_type**: 模型类型

    返回模型的详细信息，包括准确率、混淆矩阵、训练时间等
    """
    try:
        logger.debug(f"获取模型状态: model_type={model_type}")
        status_info = model_manager.get_model_status(model_type)
        return ModelStatusResponse(**status_info)

    except ValueError as e:
        logger.warning(f"获取模型状态失败: {str(e)}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get("/model/{model_type}/image", status_code=status.HTTP_200_OK)
async def get_model_image(model_type: str):
    """
    获取指定模型的混淆矩阵图片

    - **model_type**: 模型类型

    返回混淆矩阵热力图PNG图片
    """
    try:
        logger.debug(f"获取模型图片: model_type={model_type}")
        status_info = model_manager.get_model_status(model_type)
        image_path = status_info["image_path"]

        if not os.path.exists(image_path):
            logger.warning(f"图片文件不存在: {image_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"图片文件不存在: {image_path}",
            )

        logger.debug(f"返回模型图片: {image_path}")
        return FileResponse(
            image_path, media_type="image/png", filename=os.path.basename(image_path)
        )

    except ValueError as e:
        logger.warning(f"获取模型图片失败: {str(e)}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get(
    "/model/{model_type}/additional-image/{image_type}", status_code=status.HTTP_200_OK
)
async def get_model_additional_image(model_type: str, image_type: str):
    """
    获取指定模型的额外图片

    - **model_type**: 模型类型
    - **image_type**: 图片类型 (roc_curve, pr_curve, feature_importance)

    返回额外图片PNG文件
    """
    try:
        logger.debug(
            f"获取模型额外图片: model_type={model_type}, image_type={image_type}"
        )
        status_info = model_manager.get_model_status(model_type)
        additional_images = status_info.get("additional_images", {})

        if image_type not in additional_images:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"图片类型 {image_type} 不存在。可用的类型: {list(additional_images.keys())}",
            )

        image_path = additional_images[image_type]

        if not os.path.exists(image_path):
            logger.warning(f"图片文件不存在: {image_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"图片文件不存在: {image_path}",
            )

        logger.debug(f"返回模型额外图片: {image_path}")
        return FileResponse(
            image_path, media_type="image/png", filename=os.path.basename(image_path)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取模型额外图片失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取模型额外图片失败: {str(e)}",
        )


@router.get("/model/comparison-image/{image_type}", status_code=status.HTTP_200_OK)
async def get_comparison_image(image_type: str):
    """
    获取模型对比图片

    - **image_type**: 图片类型 (accuracy_comparison, training_time_comparison, radar_comparison, metrics_comparison)

    返回对比图片PNG文件
    """
    try:
        logger.debug(f"获取模型对比图片: image_type={image_type}")
        from ..models import ModelComparator

        # 生成对比图片
        comparator = ModelComparator()
        comparison_images = comparator.generate_comparison_plots(
            model_manager.model_info, model_manager.best_model_name
        )

        if image_type not in comparison_images:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"对比图片类型 {image_type} 不存在。可用的类型: {list(comparison_images.keys())}",
            )

        image_path = comparison_images[image_type]

        if not os.path.exists(image_path):
            logger.warning(f"图片文件不存在: {image_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"图片文件不存在: {image_path}",
            )

        logger.debug(f"返回模型对比图片: {image_path}")
        return FileResponse(
            image_path, media_type="image/png", filename=os.path.basename(image_path)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取模型对比图片失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取模型对比图片失败: {str(e)}",
        )


@router.get(
    "/model/all", response_model=AllModelsResponse, status_code=status.HTTP_200_OK
)
async def get_all_models():
    """
    获取所有模型的信息和对比图片

    返回：
    - 所有模型的详细信息（按准确率排序）
    - 对比图片路径（准确率对比、训练时间对比、雷达图、多指标对比）
    - 最佳模型标识
    """
    try:
        logger.debug("获取所有模型信息")
        from ..models import ModelComparator

        all_models_info = model_manager.get_all_models_info()

        # 生成对比图片
        comparator = ModelComparator()
        comparison_images = comparator.generate_comparison_plots(
            model_manager.model_info, model_manager.best_model_name
        )

        all_models_info["comparison_images"] = comparison_images

        logger.info(
            f"成功获取所有模型信息: 共 {all_models_info['total_models']} 个模型"
        )
        return AllModelsResponse(**all_models_info)

    except Exception as e:
        logger.error(f"获取所有模型信息失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取所有模型信息失败: {str(e)}",
        )


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check_endpoint():
    """
    系统健康检查（增强版）

    返回系统运行状态，包括：
    - 是否已训练模型
    - 模型是否可以正常工作
    - 系统运行时长
    - 最佳模型信息
    - 数据库连接状态
    - 磁盘空间
    - 内存使用
    - CPU使用率
    """
    try:
        logger.debug("执行健康检查")
        # 基础健康信息
        basic_health = model_manager.get_health_info()

        # 综合健康检查
        comprehensive = health_check.comprehensive_health_check()

        # 合并健康信息
        health_info = {
            **basic_health,
            "system_health": comprehensive,
            "database_connected": comprehensive["checks"]["database"]["is_connected"],
            "disk_healthy": comprehensive["checks"]["disk"]["is_healthy"],
            "memory_healthy": comprehensive["checks"]["memory"]["is_healthy"],
            "cpu_healthy": comprehensive["checks"]["cpu"]["is_healthy"],
        }

        # 更新整体状态
        if not comprehensive["all_healthy"]:
            health_info["status"] = "degraded"

        logger.info(
            f"健康检查完成: status={health_info['status']}, model_count={health_info['model_count']}"
        )
        return HealthResponse(**health_info)

    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"健康检查失败: {str(e)}",
        )


@router.get(
    "/model/{model_type}/versions",
    response_model=ModelVersionsResponse,
    status_code=status.HTTP_200_OK,
)
async def get_model_versions(model_type: str, limit: int = 50):
    """
    获取指定模型的所有版本

    - **model_type**: 模型类型
    - **limit**: 返回的最大版本数量（默认50）

    返回该模型的所有版本信息，包括版本号、准确率、创建时间等
    """
    try:
        logger.debug(f"获取模型版本列表: model_type={model_type}, limit={limit}")
        db = get_db()

        # 在会话内完成所有数据访问和转换
        with db.get_session() as session:
            from ..database.models import ModelVersion

            query = session.query(ModelVersion).filter(
                ModelVersion.model_type == model_type
            )
            versions = query.order_by(ModelVersion.created_at.desc()).limit(limit).all()

            # 找到激活版本和最佳版本
            active_version = None
            best_version = None
            version_dicts = []
            for version in versions:
                if version.is_active is True:
                    version_str = version.version
                    active_version = (
                        str(version_str) if version_str is not None else None
                    )
                if version.is_best is True:
                    version_str = version.version
                    best_version = str(version_str) if version_str is not None else None
                # 在会话内转换为字典
                version_dicts.append(version.to_dict())

        return ModelVersionsResponse(
            model_type=model_type,
            total_versions=len(version_dicts),
            versions=[ModelVersionInfo(**v) for v in version_dicts],
            active_version=active_version,
            best_version=best_version,
        )

    except Exception as e:
        logger.error(f"获取模型版本列表失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取模型版本列表失败: {str(e)}",
        )


@router.post(
    "/model/{model_type}/versions/{version}/activate",
    status_code=status.HTTP_200_OK,
)
async def activate_model_version(model_type: str, version: str):
    """
    激活指定版本的模型

    - **model_type**: 模型类型
    - **version**: 版本号

    将指定版本设置为激活状态，用于预测
    """
    try:
        logger.info(f"激活模型版本: model_type={model_type}, version={version}")
        db = get_db()
        success = db.activate_model_version(model_type, version)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"模型版本不存在: {model_type} v{version}",
            )

        return {"success": True, "message": f"模型版本 {model_type} v{version} 已激活"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"激活模型版本失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"激活模型版本失败: {str(e)}",
        )


@router.delete(
    "/model/{model_type}/versions/{version}",
    status_code=status.HTTP_200_OK,
)
async def delete_model_version(model_type: str, version: str):
    """
    删除指定版本的模型

    - **model_type**: 模型类型
    - **version**: 版本号

    注意：删除操作不可恢复，请谨慎操作
    """
    try:
        logger.warning(f"删除模型版本: model_type={model_type}, version={version}")
        db = get_db()
        success = db.delete_model_version(model_type, version)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"模型版本不存在: {model_type} v{version}",
            )

        return {"success": True, "message": f"模型版本 {model_type} v{version} 已删除"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除模型版本失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除模型版本失败: {str(e)}",
        )


@router.get(
    "/predictions/history",
    response_model=PredictionHistoryResponse,
    status_code=status.HTTP_200_OK,
)
async def get_prediction_history(
    limit: int = 100,
    offset: int = 0,
    url: Optional[str] = None,
    model_type: Optional[str] = None,
):
    """
    获取预测历史记录

    - **limit**: 返回的最大记录数（默认100）
    - **offset**: 偏移量（默认0）
    - **url**: 可选，按URL过滤
    - **model_type**: 可选，按模型类型过滤

    返回预测历史记录列表
    """
    try:
        logger.debug(
            f"获取预测历史: limit={limit}, offset={offset}, url={url}, model_type={model_type}"
        )
        db = get_db()

        # 由于 Database.get_prediction_history 不支持 model_type 过滤，我们需要手动查询
        with db.get_session() as session:
            query = session.query(PredictionHistory)
            if url:
                query = query.filter(PredictionHistory.url.contains(url))
            if model_type:
                query = query.filter(PredictionHistory.model_type == model_type)

            # 获取总数
            total = query.count()

            # 获取分页数据
            predictions = (
                query.order_by(PredictionHistory.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )

            # 在会话关闭前转换为字典
            history_items = [PredictionHistoryItem(**p.to_dict()) for p in predictions]

        return PredictionHistoryResponse(
            total=total,
            limit=limit,
            offset=offset,
            history=history_items,
        )

    except Exception as e:
        logger.error(f"获取预测历史失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取预测历史失败: {str(e)}",
        )


@router.get(
    "/predictions/stats",
    response_model=PredictionStatsResponse,
    status_code=status.HTTP_200_OK,
)
async def get_prediction_stats():
    """
    获取预测统计信息

    返回预测总数、安全/不安全预测数量、安全比例等统计信息
    """
    try:
        logger.debug("获取预测统计信息")
        db = get_db()
        stats = db.get_prediction_stats()

        return PredictionStatsResponse(**stats)

    except Exception as e:
        logger.error(f"获取预测统计失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取预测统计失败: {str(e)}",
        )


@router.get("/", status_code=status.HTTP_200_OK)
async def root():
    """根路径，返回API信息"""
    return {
        "message": "网络安全威胁检测系统API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "train": "/api/train",
            "predict": "/api/predict",
            "batch_predict": "/api/predict/batch",
            "model_status": "/api/model/{model_type}/status",
            "model_image": "/api/model/{model_type}/image",
            "model_versions": "/api/model/{model_type}/versions",
            "activate_version": "/api/model/{model_type}/versions/{version}/activate",
            "delete_version": "/api/model/{model_type}/versions/{version}",
            "all_models": "/api/model/all",
            "prediction_history": "/api/predictions/history",
            "prediction_stats": "/api/predictions/stats",
            "health": "/api/health",
        },
    }
