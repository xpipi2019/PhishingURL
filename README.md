# 网络安全威胁检测系统

基于机器学习的URL安全检测系统，包含后端API服务和前端Web界面。

## 📋 目录

- [功能特性](#功能特性)
- [技术栈](#技术栈)
- [快速开始](#快速开始)
- [安装部署](#安装部署)
- [配置说明](#配置说明)
- [API使用](#api使用)
- [前端使用](#前端使用)
- [项目结构](#项目结构)
- [故障排查](#故障排查)
- [开发指南](#开发指南)
- [许可证](#许可证)

## ✨ 功能特性

### 核心功能

- **8种机器学习模型训练**：
  - Logistic Regression（逻辑回归）
  - KNN（K近邻）
  - SVM（支持向量机）
  - Kernel SVM（核支持向量机）
  - Naive Bayes（朴素贝叶斯）
  - Decision Tree（决策树）
  - Random Forest（随机森林）
  - XGBoost

- **模型管理**：
  - 自动记录最佳模型
  - 模型版本管理
  - 模型激活/切换
  - 模型删除
  - 模型对比分析

- **预测功能**：
  - 单URL预测（自动特征提取）
  - 批量URL预测（异步处理）
  - 预测历史记录
  - 预测统计分析

- **数据管理**：
  - SQLite数据库存储
  - 预测历史持久化
  - 模型版本追踪

- **监控与健康检查**：
  - 系统健康检查
  - 数据库连接监控
  - 磁盘空间监控
  - 内存使用监控
  - CPU使用率监控

- **日志系统**：
  - 结构化JSON日志
  - 文本格式日志
  - 日志轮转和压缩
  - 错误日志单独记录

- **可视化**：
  - 混淆矩阵热力图
  - ROC曲线
  - PR曲线
  - 特征重要性图
  - 模型对比图表（准确率、训练时间、雷达图、多指标对比）

### 前端功能

- ✅ URL预测（单个和批量）
- ✅ 模型训练界面
- ✅ 模型管理（查看状态、版本管理、激活、删除）
- ✅ 预测历史查询（支持过滤和分页）
- ✅ 系统健康检查
- ✅ 可视化图表展示

## 🛠 技术栈

### 后端

- **Web框架**: FastAPI 0.109.0
- **机器学习**: scikit-learn, XGBoost
- **数据处理**: pandas, numpy
- **数据库**: SQLAlchemy (SQLite)
- **可视化**: matplotlib, seaborn
- **日志**: loguru
- **配置管理**: python-dotenv

### 前端

- **框架**: Vue 3
- **构建工具**: Vite
- **路由**: Vue Router
- **HTTP客户端**: Axios

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone <repository-url>
cd PhishingURL
```

### 2. 后端设置

```bash
# 进入后端目录
cd backend

# 安装Python依赖
pip install -r requirements.txt

# 配置环境变量（可选）
cp .env.example .env
# 编辑 .env 文件

# 启动后端服务
python main.py
```

后端服务将在 `http://localhost:8088` 启动

### 3. 前端设置

```bash
# 进入前端目录
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

前端应用将在 `http://localhost:3000` 启动

### 4. 访问应用

- **前端界面**: http://localhost:3000
- **API文档（Swagger UI）**: http://localhost:8088/docs
- **API文档（ReDoc）**: http://localhost:8088/redoc
- **健康检查**: http://localhost:8088/api/health

## 📦 安装部署

### 开发环境部署

#### 后端

1. **Python环境要求**：
   - Python 3.8+
   - pip 或 conda

2. **安装步骤**：

```bash
# 进入后端目录
cd backend

# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 启动服务
python main.py
```

#### 前端

```bash
cd frontend

# 安装依赖
npm install

# 开发模式
npm run dev

# 构建生产版本
npm run build
```

### 生产环境部署

#### 方式1：使用 Uvicorn（推荐）

```bash
# 进入后端目录
cd backend

# 安装uvicorn（如果未安装）
pip install uvicorn[standard]

# 启动服务
uvicorn main:app --host 0.0.0.0 --port 8088 --workers 4
```

#### 方式2：使用 Gunicorn + Uvicorn

```bash
# 进入后端目录
cd backend

# 安装gunicorn
pip install gunicorn

# 启动服务
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8088
```

#### 方式3：使用 Docker

创建 `Dockerfile`：

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY backend/requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制后端项目文件
COPY backend/ .

# 暴露端口
EXPOSE 8088

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8088"]
```

构建和运行：

```bash
# 构建镜像（在项目根目录执行）
docker build -t phishing-detection-api .

# 运行容器
docker run -d \
  -p 8088:8088 \
  -v $(pwd)/backend/data:/app/data \
  -v $(pwd)/backend/models:/app/models \
  -v $(pwd)/backend/db:/app/db \
  -v $(pwd)/backend/logs:/app/logs \
  --name phishing-api \
  phishing-detection-api
```

#### 前端生产部署

```bash
cd frontend

# 构建生产版本
npm run build

# 构建后的文件在 dist/ 目录
# 可以使用 nginx 或其他静态文件服务器部署
```

Nginx 配置示例：

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # 前端静态文件
    location / {
        root /path/to/frontend/dist;
        try_files $uri $uri/ /index.html;
    }

    # 后端API代理
    location /api {
        proxy_pass http://127.0.0.1:8088;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## ⚙️ 配置说明

### 环境变量配置

创建 `.env` 文件：

```env
# 应用配置
APP_NAME=网络安全威胁检测系统
APP_VERSION=1.0.0
DEBUG=False

# 服务器配置
HOST=0.0.0.0
PORT=8088

# 目录配置
MODELS_DIR=models
IMAGES_DIR=images
DATA_DIR=data
LOGS_DIR=logs
DB_DIR=db

# 数据库配置
DATABASE_URL=sqlite:///db/phishing_detection.db

# CORS配置（生产环境务必配置）
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
CORS_ALLOW_CREDENTIALS=True

# 日志配置
LOG_LEVEL=INFO
LOG_FORMAT=json  # json 或 text
LOG_RETENTION_DAYS=30
LOG_ERROR_RETENTION_DAYS=90

# 特征提取配置
FEATURE_EXTRACTION_TIMEOUT=10
DEFAULT_FEATURES_COUNT=30

# 批量预测配置
BATCH_PREDICT_MAX_SIZE=100
BATCH_PREDICT_TIMEOUT=300

# 健康检查配置
HEALTH_CHECK_DISK_THRESHOLD_GB=1.0
HEALTH_CHECK_MEMORY_THRESHOLD_MB=100
```

### 前端环境变量

在 `frontend/` 目录创建 `.env` 文件：

```env
VITE_API_BASE_URL=http://localhost:8088
```

### 重要配置项说明

1. **CORS_ORIGINS**: 生产环境必须配置允许的前端域名，不要使用 `*`
2. **LOG_FORMAT**: 
   - `text`: 适合开发环境，便于阅读
   - `json`: 适合生产环境，便于日志分析
3. **BATCH_PREDICT_MAX_SIZE**: 限制批量预测数量，防止内存溢出
4. **DATABASE_URL**: 支持SQLite和PostgreSQL（需要修改连接字符串）

## 📖 API使用

详细的API使用示例请参考 [API_EXAMPLES.md](API_EXAMPLES.md)

### 基础API

#### 1. 训练模型

```bash
POST /api/train
Content-Type: application/json

{
  "dataset_path": "data/raw/PhishingData.csv",
  "model_type": "logistic_regression",
  "test_size": 0.25,
  "random_state": 0
}
```

#### 2. 预测URL

```bash
POST /api/predict
Content-Type: application/json

{
  "url": "https://example.com",
  "model_type": null
}
```

#### 3. 批量预测

```bash
POST /api/predict/batch
Content-Type: application/json

{
  "urls": ["https://example.com", "https://test.com"],
  "model_type": null
}
```

### 完整API列表

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/train` | 训练模型 |
| POST | `/api/predict` | 预测单个URL |
| POST | `/api/predict/batch` | 批量预测URL |
| GET | `/api/model/{model_type}/status` | 获取模型状态 |
| GET | `/api/model/{model_type}/image` | 获取混淆矩阵图片 |
| GET | `/api/model/{model_type}/additional-image/{image_type}` | 获取额外图片（ROC/PR/特征重要性） |
| GET | `/api/model/comparison-image/{image_type}` | 获取对比图片 |
| GET | `/api/model/{model_type}/versions` | 获取模型版本列表 |
| POST | `/api/model/{model_type}/versions/{version}/activate` | 激活模型版本 |
| DELETE | `/api/model/{model_type}/versions/{version}` | 删除模型版本 |
| GET | `/api/model/all` | 获取所有模型信息 |
| GET | `/api/predictions/history` | 获取预测历史 |
| GET | `/api/predictions/stats` | 获取预测统计 |
| GET | `/api/health` | 健康检查 |

## 🖥 前端使用

### 功能说明

1. **URL预测**
   - 支持单个URL预测
   - 支持批量URL预测（每行一个URL）
   - 可选择使用的模型类型
   - 显示预测结果和概率分布

2. **模型训练**
   - 选择模型类型（8种可选）
   - 配置数据集路径
   - 设置测试集比例和随机种子
   - 查看训练结果和评估指标

3. **模型管理**
   - 查看所有模型列表
   - 查看模型详情和可视化图表
   - 版本管理（查看、激活、删除）
   - 模型对比图表展示

4. **预测历史**
   - 查看历史预测记录
   - 支持URL和模型类型过滤
   - 分页显示
   - 统计信息展示

5. **健康检查**
   - 系统运行状态
   - 数据库、磁盘、内存、CPU监控
   - 自动刷新（30秒）

### 开发命令

```bash
cd frontend

# 安装依赖
npm install

# 开发模式
npm run dev

# 构建生产版本
npm run build

# 预览生产构建
npm run preview
```

## 📁 项目结构

```
PhishingURL/
├── README.md                    # 项目说明文档（本文件）
├── API_EXAMPLES.md              # API使用示例文档
├── UPGRADE_GUIDE.md             # 升级指南
├── CHANGELOG.md                 # 更新日志
├── .env.example                 # 环境变量示例文件
├── .gitignore                   # Git忽略文件配置
│
├── backend/                     # 后端代码目录
│   ├── main.py                  # FastAPI主应用入口
│   ├── requirements.txt         # Python依赖包列表
│   │
│   ├── src/                     # 后端源代码目录
│   ├── __init__.py
│   ├── config.py                # 配置管理模块
│   ├── constants.py             # 常量定义
│   │
│   ├── api/                     # API模块
│   │   ├── __init__.py
│   │   ├── routes.py            # API路由定义
│   │   └── schemas.py           # Pydantic数据模型
│   │
│   ├── data/                    # 数据处理模块
│   │   ├── __init__.py
│   │   ├── data_loader.py       # 数据加载器
│   │   ├── data_preprocessor.py # 数据预处理器
│   │   ├── feature_extractor.py # URL特征提取器
│   │   └── domain_config.json   # 域名配置
│   │
│   ├── models/                  # 模型模块
│   │   ├── __init__.py
│   │   ├── model_trainer.py     # 模型训练器
│   │   ├── model_predictor.py   # 模型预测器
│   │   ├── model_manager.py     # 模型管理器
│   │   ├── model_evaluator.py   # 模型评估器
│   │   └── model_comparator.py  # 模型对比器
│   │
│   ├── database/                # 数据库模块
│   │   ├── __init__.py
│   │   ├── models.py            # 数据库模型定义
│   │   └── database.py          # 数据库操作
│   │
│   └── utils/                   # 工具模块
│       ├── __init__.py
│       ├── logger.py            # 日志工具
│       └── health_check.py      # 健康检查工具
│   │
│   ├── models/                  # 模型文件存储目录（自动创建）
│   ├── images/                  # 可视化图片存储目录（自动创建）
│   ├── data/                    # 数据文件目录
│   │   ├── raw/                 # 原始数据
│   │   └── processed/           # 处理后的数据
│   ├── logs/                    # 日志文件目录（自动创建）
│   └── db/                      # 数据库文件目录（自动创建）
│
├── frontend/                    # 前端源代码目录
│   ├── src/
│   │   ├── api/                 # API服务层
│   │   │   ├── client.js        # Axios客户端配置
│   │   │   └── services.js      # API服务函数
│   │   ├── views/               # 页面组件
│   │   │   ├── PredictView.vue  # URL预测页面
│   │   │   ├── TrainView.vue    # 模型训练页面
│   │   │   ├── ModelsView.vue   # 模型管理页面
│   │   │   ├── HistoryView.vue   # 预测历史页面
│   │   │   └── HealthView.vue   # 健康检查页面
│   │   ├── App.vue              # 根组件
│   │   ├── main.js              # 入口文件
│   │   └── style.css            # 全局样式
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── README.md
```

## 🔧 故障排查

### 常见问题

#### 1. 服务启动失败

**问题**: `ModuleNotFoundError` 或导入错误

**解决方案**:
```bash
# 检查Python版本
python --version  # 需要3.8+

# 重新安装依赖
pip install -r requirements.txt

# 检查虚拟环境
which python  # 确认使用的是正确的Python环境
```

#### 2. 数据库连接失败

**问题**: `sqlite3.OperationalError` 或数据库文件无法创建

**解决方案**:
```bash
# 进入后端目录
cd backend

# 检查db目录权限
mkdir -p db
chmod 755 db

# 检查数据库URL配置
# 在.env文件中确认 DATABASE_URL 配置正确

# 手动初始化数据库
python -c "from src.database import init_db; init_db()"
```

#### 3. 前端无法连接后端

**问题**: CORS错误或连接失败

**解决方案**:
```bash
# 检查后端服务是否运行
curl http://localhost:8088/api/health

# 检查前端API配置
# 在 frontend/.env 中确认 VITE_API_BASE_URL 配置正确

# 检查CORS配置
# 在 .env 中确认 CORS_ORIGINS 包含前端地址
```

#### 4. 模型训练失败

**问题**: 训练时出现错误

**可能原因和解决方案**:

- **数据集文件不存在**:
  ```bash
  # 检查文件路径
  ls -l data/raw/PhishingData.csv
  
  # 使用绝对路径
  {
    "dataset_path": "/absolute/path/to/data/raw/PhishingData.csv"
  }
  ```

- **数据集格式错误**:
  - 确保CSV文件格式正确
  - 最后一列必须是标签列
  - 特征列的值必须在 -1, 0, 1 范围内

- **内存不足**:
  - 减少测试集比例: `test_size: 0.2`
  - 使用较小的数据集
  - 增加系统内存

#### 5. 预测失败

**问题**: 预测时返回错误

**可能原因和解决方案**:

- **没有训练模型**:
  ```bash
  # 先训练模型
  POST /api/train
  ```

- **特征数量不匹配**:
  - 确保使用的模型与训练时的特征数量一致
  - 检查 `DEFAULT_FEATURES_COUNT` 配置

- **模型文件损坏**:
  ```bash
  # 重新训练模型
  POST /api/train
  ```

### 日志查看

```bash
# 进入后端目录
cd backend

# 进入后端目录
cd backend

# 查看应用日志
tail -f logs/app_$(date +%Y-%m-%d).log

# 查看错误日志
tail -f logs/error_$(date +%Y-%m-%d).log

# 如果是JSON格式，使用jq格式化
cat logs/app_2025-12-25.log | jq .
```

## 💻 开发指南

### 代码规范

- 使用类型提示（Type Hints）
- 遵循PEP 8代码风格
- 添加详细的文档字符串
- 使用常量而非魔法数字

### 添加新模型类型

1. 在 `backend/src/models/model_trainer.py` 的 `MODEL_TYPES` 中添加：

```python
MODEL_TYPES = {
    # ... 现有模型
    "new_model": NewModelClass,
}
```

2. 在 `_get_model_classifier` 方法中添加初始化逻辑

3. 更新 `backend/src/api/schemas.py` 中的模型类型示例

4. 更新 `backend/src/constants.py` 中的 `SUPPORTED_MODEL_TYPES`

### 添加新API端点

1. 在 `backend/src/api/routes.py` 中添加路由函数

2. 在 `backend/src/api/schemas.py` 中定义请求/响应模型

3. 更新 `backend/src/api/__init__.py` 导出新的schemas

### 测试

```bash
# 运行健康检查
curl http://localhost:8088/api/health

# 测试预测（需要先训练模型）
curl -X POST http://localhost:8088/api/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

## 📚 相关文档

- [API使用示例](API_EXAMPLES.md) - 详细的API使用示例和代码
- [升级指南](UPGRADE_GUIDE.md) - 版本升级和迁移指南
- [更新日志](CHANGELOG.md) - 版本更新记录
- [特征提取说明](HowToGetFeat.md) - URL特征提取详细说明
- [前端README](frontend/README.md) - 前端项目说明

## 🔒 安全建议

1. **生产环境配置**:
   - 设置 `DEBUG=False`
   - 配置 `CORS_ORIGINS` 为具体域名
   - 使用HTTPS

2. **数据库安全**:
   - 定期备份数据库
   - 使用强密码（如果使用PostgreSQL）
   - 限制数据库访问权限

3. **API安全**:
   - 考虑添加API认证（JWT或API Key）
   - 实施速率限制
   - 验证和清理输入数据

## 📊 性能指标

- **单次预测响应时间**: < 100ms（不含特征提取）
- **批量预测（100个URL）**: < 5秒
- **模型训练时间**: 取决于数据集大小和模型类型
- **并发支持**: 支持异步处理，可处理多个并发请求

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📝 更新日志

详细的更新日志请查看 [CHANGELOG.md](CHANGELOG.md)

## 📄 许可证

MIT License

## 🙏 致谢

- FastAPI - 现代、快速的Web框架
- Vue.js - 渐进式JavaScript框架
- scikit-learn - 机器学习库
- XGBoost - 梯度提升框架
- SQLAlchemy - Python SQL工具包

## 📞 支持

如有问题或建议，请：
- 查看 [故障排查](#故障排查) 部分
- 查看 [API_EXAMPLES.md](API_EXAMPLES.md) 获取使用示例
- 提交 Issue 或 Pull Request
