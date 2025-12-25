# 网络安全威胁检测系统 - 前端

基于 Vue 3 + Vite 的前端应用，提供简洁的用户界面来使用后端API。

## 功能特性

- ✅ URL预测（单个和批量）
- ✅ 模型训练
- ✅ 模型管理（查看状态、版本管理、激活、删除）
- ✅ 预测历史查询
- ✅ 系统健康检查

## 快速开始

### 安装依赖

```bash
npm install
```

### 开发模式

```bash
npm run dev
```

应用将在 http://localhost:3000 启动

### 构建生产版本

```bash
npm run build
```

### 预览生产构建

```bash
npm run preview
```

## 配置

默认API地址为 `http://localhost:8088`，可以通过环境变量修改：

创建 `.env` 文件：

```env
VITE_API_BASE_URL=http://localhost:8088
```

## 项目结构

```
frontend/
├── src/
│   ├── api/           # API服务层
│   │   ├── client.js  # Axios客户端配置
│   │   └── services.js # API服务函数
│   ├── views/         # 页面组件
│   │   ├── PredictView.vue    # URL预测页面
│   │   ├── TrainView.vue      # 模型训练页面
│   │   ├── ModelsView.vue     # 模型管理页面
│   │   ├── HistoryView.vue     # 预测历史页面
│   │   └── HealthView.vue     # 健康检查页面
│   ├── App.vue        # 根组件
│   ├── main.js        # 入口文件
│   └── style.css      # 全局样式
├── index.html
├── package.json
├── vite.config.js
└── README.md
```

## 使用说明

1. **URL预测**: 输入单个URL或批量URL进行预测
2. **模型训练**: 选择模型类型和数据集路径进行训练
3. **模型管理**: 查看所有模型信息、版本管理、激活/删除版本
4. **预测历史**: 查看历史预测记录，支持过滤和分页
5. **健康检查**: 查看系统运行状态和健康指标

## 注意事项

- 确保后端服务已启动（默认端口8088）
- 模型训练可能需要较长时间，请耐心等待
- 批量预测建议不超过100个URL

