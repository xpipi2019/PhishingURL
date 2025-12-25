<template>
  <div>
    <div class="card">
      <h2>模型管理</h2>
      <button class="btn btn-secondary" @click="loadAllModels" :disabled="loading">
        {{ loading ? '加载中...' : '刷新模型列表' }}
      </button>
    </div>

    <!-- 所有模型列表 -->
    <div v-if="allModels" class="card">
      <h2>所有模型 (共 {{ allModels.total_models }} 个)</h2>
      <div v-if="allModels.best_model" class="form-group">
        <strong>最佳模型:</strong> 
        <span class="badge badge-success">{{ allModels.best_model }}</span>
        <span style="margin-left: 1rem;">准确率: {{ (allModels.best_accuracy * 100).toFixed(2) }}%</span>
      </div>

      <!-- 模型对比图片 -->
      <div v-if="allModels.comparison_images && Object.keys(allModels.comparison_images).length > 0" class="form-group" style="margin-top: 1.5rem;">
        <h3>模型对比图表</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 1rem; margin-top: 1rem;">
          <div v-if="allModels.comparison_images.accuracy_comparison">
            <h4 style="margin-bottom: 0.5rem;">准确率对比</h4>
            <img 
              :src="getComparisonImageUrl('accuracy_comparison')" 
              alt="准确率对比" 
              style="width: 100%; border: 1px solid #ddd; border-radius: 4px;"
            />
          </div>
          <div v-if="allModels.comparison_images.training_time_comparison">
            <h4 style="margin-bottom: 0.5rem;">训练时间对比</h4>
            <img 
              :src="getComparisonImageUrl('training_time_comparison')" 
              alt="训练时间对比" 
              style="width: 100%; border: 1px solid #ddd; border-radius: 4px;"
            />
          </div>
          <div v-if="allModels.comparison_images.radar_comparison">
            <h4 style="margin-bottom: 0.5rem;">雷达图对比</h4>
            <img 
              :src="getComparisonImageUrl('radar_comparison')" 
              alt="雷达图对比" 
              style="width: 100%; border: 1px solid #ddd; border-radius: 4px;"
            />
          </div>
          <div v-if="allModels.comparison_images.metrics_comparison">
            <h4 style="margin-bottom: 0.5rem;">多指标对比</h4>
            <img 
              :src="getComparisonImageUrl('metrics_comparison')" 
              alt="多指标对比" 
              style="width: 100%; border: 1px solid #ddd; border-radius: 4px;"
            />
          </div>
        </div>
      </div>
      
      <table class="table" style="margin-top: 1rem;">
        <thead>
          <tr>
            <th>模型类型</th>
            <th>准确率</th>
            <th>训练时间(秒)</th>
            <th>训练样本</th>
            <th>测试样本</th>
            <th>特征数</th>
            <th>状态</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="model in allModels.models" :key="model.model_type">
            <td>{{ model.model_type }}</td>
            <td>
              <strong style="color: #27ae60;">{{ (model.accuracy * 100).toFixed(2) }}%</strong>
            </td>
            <td>{{ model.training_time.toFixed(2) }}</td>
            <td>{{ model.train_samples }}</td>
            <td>{{ model.test_samples }}</td>
            <td>{{ model.features_count }}</td>
            <td>
              <span v-if="model.is_best_model" class="badge badge-success">最佳</span>
              <span v-else class="badge badge-info">已训练</span>
            </td>
            <td>
              <button 
                class="btn btn-secondary" 
                style="padding: 0.25rem 0.5rem; font-size: 0.875rem;"
                @click="viewModelDetails(model.model_type)"
              >
                查看详情
              </button>
              <button 
                class="btn btn-secondary" 
                style="padding: 0.25rem 0.5rem; font-size: 0.875rem; margin-left: 0.5rem;"
                @click="viewModelVersions(model.model_type)"
              >
                版本管理
              </button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- 模型详情 -->
    <div v-if="modelStatus" class="card">
      <h2>模型详情: {{ modelStatus.model_type }}</h2>
      <div class="form-group">
        <strong>准确率:</strong> {{ (modelStatus.accuracy * 100).toFixed(2) }}%
      </div>
      <div class="form-group">
        <strong>训练时间:</strong> {{ modelStatus.training_time.toFixed(2) }} 秒
      </div>
      <div class="form-group">
        <strong>训练样本:</strong> {{ modelStatus.train_samples }}
      </div>
      <div class="form-group">
        <strong>测试样本:</strong> {{ modelStatus.test_samples }}
      </div>
      <div class="form-group">
        <strong>特征数量:</strong> {{ modelStatus.features_count }}
      </div>
      <div class="form-group">
        <strong>创建时间:</strong> {{ modelStatus.created_at }}
      </div>
      <div class="form-group" v-if="modelStatus.image_exists">
        <strong>混淆矩阵图:</strong>
        <div style="margin-top: 1rem;">
          <img 
            :src="getModelImageUrl(modelStatus.model_type)" 
            alt="混淆矩阵" 
            style="max-width: 100%; border: 1px solid #ddd; border-radius: 4px;"
          />
        </div>
      </div>

      <!-- 额外图片 -->
      <div v-if="modelStatus.additional_images && Object.keys(modelStatus.additional_images).length > 0" class="form-group" style="margin-top: 1.5rem;">
        <h3>额外图表</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 1rem; margin-top: 1rem;">
          <div v-if="modelStatus.additional_images.roc_curve">
            <h4 style="margin-bottom: 0.5rem;">ROC曲线</h4>
            <img 
              :src="getModelAdditionalImageUrl(modelStatus.model_type, 'roc_curve')" 
              alt="ROC曲线" 
              style="width: 100%; border: 1px solid #ddd; border-radius: 4px;"
            />
          </div>
          <div v-if="modelStatus.additional_images.pr_curve">
            <h4 style="margin-bottom: 0.5rem;">PR曲线</h4>
            <img 
              :src="getModelAdditionalImageUrl(modelStatus.model_type, 'pr_curve')" 
              alt="PR曲线" 
              style="width: 100%; border: 1px solid #ddd; border-radius: 4px;"
            />
          </div>
          <div v-if="modelStatus.additional_images.feature_importance">
            <h4 style="margin-bottom: 0.5rem;">特征重要性</h4>
            <img 
              :src="getModelAdditionalImageUrl(modelStatus.model_type, 'feature_importance')" 
              alt="特征重要性" 
              style="width: 100%; border: 1px solid #ddd; border-radius: 4px;"
            />
          </div>
        </div>
      </div>
    </div>

    <!-- 模型版本列表 -->
    <div v-if="modelVersions" class="card">
      <h2>模型版本: {{ modelVersions.model_type }} (共 {{ modelVersions.total_versions }} 个版本)</h2>
      <div class="form-group" v-if="modelVersions.active_version">
        <strong>激活版本:</strong> 
        <span class="badge badge-success">{{ modelVersions.active_version }}</span>
      </div>
      <div class="form-group" v-if="modelVersions.best_version">
        <strong>最佳版本:</strong> 
        <span class="badge badge-warning">{{ modelVersions.best_version }}</span>
      </div>
      
      <table class="table" style="margin-top: 1rem;">
        <thead>
          <tr>
            <th>版本</th>
            <th>准确率</th>
            <th>训练时间</th>
            <th>创建时间</th>
            <th>状态</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="version in modelVersions.versions" :key="version.id">
            <td>{{ version.version }}</td>
            <td>{{ (version.accuracy * 100).toFixed(2) }}%</td>
            <td>{{ version.training_time.toFixed(2) }}秒</td>
            <td>{{ version.created_at }}</td>
            <td>
              <span v-if="version.is_active" class="badge badge-success">激活</span>
              <span v-if="version.is_best" class="badge badge-warning">最佳</span>
            </td>
            <td>
              <button 
                v-if="!version.is_active"
                class="btn btn-success" 
                style="padding: 0.25rem 0.5rem; font-size: 0.875rem;"
                @click="activateVersion(modelVersions.model_type, version.version)"
              >
                激活
              </button>
              <button 
                class="btn btn-danger" 
                style="padding: 0.25rem 0.5rem; font-size: 0.875rem; margin-left: 0.5rem;"
                @click="deleteVersion(modelVersions.model_type, version.version)"
              >
                删除
              </button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- 消息提示 -->
    <div v-if="message" :class="['alert', messageType]">
      {{ message }}
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'
import { 
  getAllModels, 
  getModelStatus, 
  getModelVersions, 
  activateModelVersion, 
  deleteModelVersion,
  getModelImage,
  getModelAdditionalImage,
  getComparisonImage
} from '../api/services'

export default {
  name: 'ModelsView',
  setup() {
    const loading = ref(false)
    const allModels = ref(null)
    const modelStatus = ref(null)
    const modelVersions = ref(null)
    const message = ref('')
    const messageType = ref('')

    const showMessage = (msg, type = 'info') => {
      message.value = msg
      messageType.value = `alert-${type}`
      setTimeout(() => {
        message.value = ''
      }, 5000)
    }

    const loadAllModels = async () => {
      loading.value = true
      try {
        allModels.value = await getAllModels()
        modelStatus.value = null
        modelVersions.value = null
      } catch (error) {
        showMessage(error.message || '加载模型列表失败', 'error')
      } finally {
        loading.value = false
      }
    }

    const viewModelDetails = async (modelType) => {
      loading.value = true
      try {
        modelStatus.value = await getModelStatus(modelType)
        modelVersions.value = null
      } catch (error) {
        showMessage(error.message || '加载模型详情失败', 'error')
      } finally {
        loading.value = false
      }
    }

    const viewModelVersions = async (modelType) => {
      loading.value = true
      try {
        modelVersions.value = await getModelVersions(modelType)
        modelStatus.value = null
      } catch (error) {
        showMessage(error.message || '加载模型版本失败', 'error')
      } finally {
        loading.value = false
      }
    }

    const activateVersion = async (modelType, version) => {
      if (!confirm(`确定要激活版本 ${version} 吗？`)) {
        return
      }

      try {
        const result = await activateModelVersion(modelType, version)
        showMessage(result.message || '激活成功', 'success')
        if (modelVersions.value) {
          await viewModelVersions(modelType)
        }
      } catch (error) {
        showMessage(error.message || '激活失败', 'error')
      }
    }

    const deleteVersion = async (modelType, version) => {
      if (!confirm(`确定要删除版本 ${version} 吗？此操作不可恢复！`)) {
        return
      }

      try {
        const result = await deleteModelVersion(modelType, version)
        showMessage(result.message || '删除成功', 'success')
        if (modelVersions.value) {
          await viewModelVersions(modelType)
        }
        if (allModels.value) {
          await loadAllModels()
        }
      } catch (error) {
        showMessage(error.message || '删除失败', 'error')
      }
    }

    const getModelImageUrl = (modelType) => {
      return getModelImage(modelType)
    }

    const getModelAdditionalImageUrl = (modelType, imageType) => {
      return getModelAdditionalImage(modelType, imageType)
    }

    const getComparisonImageUrl = (imageType) => {
      return getComparisonImage(imageType)
    }

    onMounted(() => {
      loadAllModels()
    })

    return {
      loading,
      allModels,
      modelStatus,
      modelVersions,
      message,
      messageType,
      loadAllModels,
      viewModelDetails,
      viewModelVersions,
      activateVersion,
      deleteVersion,
      getModelImageUrl,
      getModelAdditionalImageUrl,
      getComparisonImageUrl
    }
  }
}
</script>

