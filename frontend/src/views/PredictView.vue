<template>
  <div>
    <div class="card">
      <h2>URL预测</h2>
      
      <div class="form-group">
        <label>预测模式</label>
        <select v-model="mode" @change="onModeChange">
          <option value="single">单个URL</option>
          <option value="batch">批量URL</option>
        </select>
      </div>

      <!-- 单个URL预测 -->
      <div v-if="mode === 'single'">
        <div class="form-group">
          <label>URL地址</label>
          <input 
            v-model="singleUrl" 
            type="text" 
            placeholder="https://example.com"
            @keyup.enter="handleSinglePredict"
          />
        </div>
        <div class="form-group">
          <label>模型类型（可选，默认使用最佳模型）</label>
          <select v-model="selectedModel">
            <option value="">使用最佳模型</option>
            <option v-for="model in availableModels" :key="model" :value="model">
              {{ model }}
            </option>
          </select>
        </div>
        <button class="btn btn-primary" @click="handleSinglePredict" :disabled="loading || !singleUrl">
          {{ loading ? '预测中...' : '开始预测' }}
        </button>
      </div>

      <!-- 批量URL预测 -->
      <div v-if="mode === 'batch'">
        <div class="form-group">
          <label>URL列表（每行一个）</label>
          <textarea 
            v-model="batchUrls" 
            placeholder="https://example.com&#10;https://test.com&#10;https://demo.com"
            rows="10"
          ></textarea>
        </div>
        <div class="form-group">
          <label>模型类型（可选，默认使用最佳模型）</label>
          <select v-model="selectedModel">
            <option value="">使用最佳模型</option>
            <option v-for="model in availableModels" :key="model" :value="model">
              {{ model }}
            </option>
          </select>
        </div>
        <button class="btn btn-primary" @click="handleBatchPredict" :disabled="loading || !batchUrls">
          {{ loading ? '预测中...' : '开始批量预测' }}
        </button>
      </div>

      <!-- 消息提示 -->
      <div v-if="message" :class="['alert', messageType]">
        {{ message }}
      </div>
    </div>

    <!-- 单个预测结果 -->
    <div v-if="singleResult" class="card">
      <h2>预测结果</h2>
      <div class="form-group">
        <strong>URL:</strong> {{ singleResult.url }}
      </div>
      <div class="form-group">
        <strong>预测结果:</strong> 
        <span :class="['badge', singleResult.is_safe ? 'badge-success' : 'badge-danger']">
          {{ singleResult.is_safe ? '安全' : '不安全' }}
        </span>
        ({{ singleResult.prediction }})
      </div>
      <div class="form-group">
        <strong>概率分布:</strong>
        <div style="margin-top: 0.5rem;">
          <span v-for="(prob, key) in singleResult.probabilities" :key="key" style="margin-right: 1rem;">
            {{ key }}: {{ (prob * 100).toFixed(2) }}%
          </span>
        </div>
      </div>
      <div class="form-group">
        <strong>使用的模型:</strong> {{ singleResult.model_used }}
      </div>
    </div>

    <!-- 批量预测结果 -->
    <div v-if="batchResults.length > 0" class="card">
      <h2>批量预测结果 (共 {{ batchResults.length }} 条)</h2>
      <table class="table">
        <thead>
          <tr>
            <th>序号</th>
            <th>URL</th>
            <th>预测结果</th>
            <th>概率</th>
            <th>状态</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(result, index) in batchResults" :key="index">
            <td>{{ index + 1 }}</td>
            <td style="max-width: 300px; word-break: break-all;">{{ result.url }}</td>
            <td>{{ result.prediction }}</td>
            <td>
              <div v-for="(prob, key) in result.probabilities" :key="key">
                {{ key }}: {{ (prob * 100).toFixed(2) }}%
              </div>
            </td>
            <td>
              <span :class="['badge', result.is_safe ? 'badge-success' : 'badge-danger']">
                {{ result.is_safe ? '安全' : '不安全' }}
              </span>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'
import { predictUrl, predictBatch, getAllModels } from '../api/services'

export default {
  name: 'PredictView',
  setup() {
    const mode = ref('single')
    const singleUrl = ref('')
    const batchUrls = ref('')
    const selectedModel = ref('')
    const loading = ref(false)
    const message = ref('')
    const messageType = ref('')
    const singleResult = ref(null)
    const batchResults = ref([])
    const availableModels = ref([])

    const loadAvailableModels = async () => {
      try {
        const data = await getAllModels()
        availableModels.value = data.models.map(m => m.model_type)
      } catch (error) {
        console.error('加载模型列表失败:', error)
      }
    }

    const showMessage = (msg, type = 'info') => {
      message.value = msg
      messageType.value = `alert-${type}`
      setTimeout(() => {
        message.value = ''
      }, 5000)
    }

    const handleSinglePredict = async () => {
      if (!singleUrl.value.trim()) {
        showMessage('请输入URL', 'error')
        return
      }

      loading.value = true
      singleResult.value = null
      message.value = ''

      try {
        const data = await predictUrl({
          url: singleUrl.value,
          model_type: selectedModel.value || null
        })
        singleResult.value = data
        showMessage('预测完成', 'success')
      } catch (error) {
        showMessage(error.message || '预测失败', 'error')
      } finally {
        loading.value = false
      }
    }

    const handleBatchPredict = async () => {
      const urls = batchUrls.value.split('\n').filter(url => url.trim())
      if (urls.length === 0) {
        showMessage('请输入至少一个URL', 'error')
        return
      }

      loading.value = true
      batchResults.value = []
      message.value = ''

      try {
        const data = await predictBatch({
          urls: urls,
          model_type: selectedModel.value || null
        })
        batchResults.value = data.results
        showMessage(`批量预测完成，共 ${data.total} 条`, 'success')
      } catch (error) {
        showMessage(error.message || '批量预测失败', 'error')
      } finally {
        loading.value = false
      }
    }

    const onModeChange = () => {
      singleResult.value = null
      batchResults.value = []
      message.value = ''
    }

    onMounted(() => {
      loadAvailableModels()
    })

    return {
      mode,
      singleUrl,
      batchUrls,
      selectedModel,
      loading,
      message,
      messageType,
      singleResult,
      batchResults,
      availableModels,
      handleSinglePredict,
      handleBatchPredict,
      onModeChange
    }
  }
}
</script>

