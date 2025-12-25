import client from './client'

// 模型训练
export const trainModel = (data) => {
  return client.post('/api/train', data)
}

// 预测单个URL
export const predictUrl = (data) => {
  return client.post('/api/predict', data)
}

// 批量预测URL
export const predictBatch = (data) => {
  return client.post('/api/predict/batch', data)
}

// 获取模型状态
export const getModelStatus = (modelType) => {
  return client.get(`/api/model/${modelType}/status`)
}

// 获取模型图片
export const getModelImage = (modelType) => {
  return `${client.defaults.baseURL}/api/model/${modelType}/image`
}

// 获取模型额外图片
export const getModelAdditionalImage = (modelType, imageType) => {
  return `${client.defaults.baseURL}/api/model/${modelType}/additional-image/${imageType}`
}

// 获取模型对比图片
export const getComparisonImage = (imageType) => {
  return `${client.defaults.baseURL}/api/model/comparison-image/${imageType}`
}

// 获取所有模型
export const getAllModels = () => {
  return client.get('/api/model/all')
}

// 获取模型版本列表
export const getModelVersions = (modelType, limit = 50) => {
  return client.get(`/api/model/${modelType}/versions`, {
    params: { limit }
  })
}

// 激活模型版本
export const activateModelVersion = (modelType, version) => {
  return client.post(`/api/model/${modelType}/versions/${version}/activate`)
}

// 删除模型版本
export const deleteModelVersion = (modelType, version) => {
  return client.delete(`/api/model/${modelType}/versions/${version}`)
}

// 获取预测历史
export const getPredictionHistory = (params) => {
  return client.get('/api/predictions/history', { params })
}

// 获取预测统计
export const getPredictionStats = () => {
  return client.get('/api/predictions/stats')
}

// 健康检查
export const healthCheck = () => {
  return client.get('/api/health')
}

