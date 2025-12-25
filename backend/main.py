"""
FastAPI主应用入口

网络安全威胁检测系统后端服务
提供基于机器学习的URL安全检测API服务

功能：
- 模型训练和管理
- URL安全预测（单个/批量）
- 预测历史记录
- 系统健康监控
- 可视化图表生成

作者: PhishingURL Team
版本: 1.0.0
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import router
from src.config import settings
from src.utils import logger

# 创建FastAPI应用实例
logger.info("正在启动FastAPI应用...")
app = FastAPI(
    title=settings.APP_NAME,
    description="基于机器学习的URL安全检测API服务，支持8种机器学习模型",
    version=settings.APP_VERSION,
    docs_url="/docs",  # Swagger UI文档
    redoc_url="/redoc",  # ReDoc文档
)

# 配置CORS中间件，允许跨域请求
# 生产环境建议在.env中配置具体的允许来源，避免使用"*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if settings.CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# 注册API路由
app.include_router(router)
logger.info("FastAPI应用初始化完成")


if __name__ == "__main__":
    """
    直接运行此文件时启动开发服务器
    
    生产环境建议使用：
    - uvicorn main:app --host 0.0.0.0 --port 8088 --workers 4
    - gunicorn + uvicorn workers
    """
    import uvicorn

    logger.info(
        f"启动Uvicorn开发服务器: host={settings.HOST}, port={settings.PORT}"
    )
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
