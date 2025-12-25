<template>
  <div>
    <div class="card">
      <h2>预测历史</h2>
      
      <div class="form-row">
        <div class="form-group">
          <label>URL过滤（可选）</label>
          <input 
            v-model="filters.url" 
            type="text" 
            placeholder="输入URL关键词"
            @keyup.enter="loadHistory"
          />
        </div>
        <div class="form-group">
          <label>模型类型过滤（可选）</label>
          <select v-model="filters.model_type">
            <option value="">全部</option>
            <option v-for="model in availableModels" :key="model" :value="model">
              {{ model }}
            </option>
          </select>
        </div>
      </div>

      <div class="form-row">
        <div class="form-group">
          <label>每页显示</label>
          <input 
            v-model.number="filters.limit" 
            type="number" 
            min="10" 
            max="500"
          />
        </div>
        <div class="form-group">
          <label>页码</label>
          <input 
            v-model.number="currentPage" 
            type="number" 
            min="1"
            @change="loadHistory"
          />
        </div>
      </div>

      <button class="btn btn-primary" @click="loadHistory" :disabled="loading">
        {{ loading ? '加载中...' : '查询' }}
      </button>
    </div>

    <!-- 统计信息 -->
    <div v-if="stats" class="stats-grid">
      <div class="stat-card">
        <h3>{{ stats.total_predictions }}</h3>
        <p>总预测数</p>
      </div>
      <div class="stat-card">
        <h3>{{ stats.safe_predictions }}</h3>
        <p>安全预测</p>
      </div>
      <div class="stat-card">
        <h3>{{ stats.unsafe_predictions }}</h3>
        <p>不安全预测</p>
      </div>
      <div class="stat-card">
        <h3>{{ (stats.safe_ratio * 100).toFixed(2) }}%</h3>
        <p>安全比例</p>
      </div>
    </div>

    <!-- 历史记录列表 -->
    <div v-if="history" class="card">
      <h2>历史记录 (共 {{ history.total }} 条)</h2>
      
      <div style="margin: 1rem 0;">
        <button 
          class="btn btn-secondary" 
          @click="prevPage" 
          :disabled="currentPage <= 1"
        >
          上一页
        </button>
        <span style="margin: 0 1rem;">
          第 {{ currentPage }} 页，共 {{ totalPages }} 页
        </span>
        <button 
          class="btn btn-secondary" 
          @click="nextPage" 
          :disabled="currentPage >= totalPages"
        >
          下一页
        </button>
      </div>

      <table class="table">
        <thead>
          <tr>
            <th>ID</th>
            <th>URL</th>
            <th>模型类型</th>
            <th>版本</th>
            <th>预测结果</th>
            <th>概率</th>
            <th>响应时间</th>
            <th>创建时间</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="item in history.history" :key="item.id">
            <td>{{ item.id }}</td>
            <td style="max-width: 300px; word-break: break-all;">{{ item.url }}</td>
            <td>{{ item.model_type }}</td>
            <td>{{ item.model_version || '-' }}</td>
            <td>
              <span :class="['badge', item.is_safe ? 'badge-success' : 'badge-danger']">
                {{ item.is_safe ? '安全' : '不安全' }}
              </span>
              <span style="margin-left: 0.5rem;">{{ item.prediction }}</span>
            </td>
            <td>
              <div v-if="item.probabilities">
                <div v-for="(prob, key) in item.probabilities" :key="key" style="font-size: 0.875rem;">
                  {{ key }}: {{ (prob * 100).toFixed(1) }}%
                </div>
              </div>
              <span v-else>-</span>
            </td>
            <td>{{ item.response_time_ms ? item.response_time_ms.toFixed(2) + 'ms' : '-' }}</td>
            <td>{{ item.created_at }}</td>
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
import { ref, computed, onMounted } from 'vue'
import { getPredictionHistory, getPredictionStats, getAllModels } from '../api/services'

export default {
  name: 'HistoryView',
  setup() {
    const loading = ref(false)
    const history = ref(null)
    const stats = ref(null)
    const message = ref('')
    const messageType = ref('')
    const currentPage = ref(1)
    const availableModels = ref([])
    const filters = ref({
      url: '',
      model_type: '',
      limit: 50
    })

    const totalPages = computed(() => {
      if (!history.value) return 1
      return Math.ceil(history.value.total / filters.value.limit)
    })

    const showMessage = (msg, type = 'info') => {
      message.value = msg
      messageType.value = `alert-${type}`
      setTimeout(() => {
        message.value = ''
      }, 5000)
    }

    const loadHistory = async () => {
      loading.value = true
      try {
        const offset = (currentPage.value - 1) * filters.value.limit
        const params = {
          limit: filters.value.limit,
          offset: offset
        }
        if (filters.value.url) {
          params.url = filters.value.url
        }
        if (filters.value.model_type) {
          params.model_type = filters.value.model_type
        }
        
        history.value = await getPredictionHistory(params)
      } catch (error) {
        showMessage(error.message || '加载历史记录失败', 'error')
      } finally {
        loading.value = false
      }
    }

    const loadStats = async () => {
      try {
        stats.value = await getPredictionStats()
      } catch (error) {
        console.error('加载统计信息失败:', error)
      }
    }

    const loadAvailableModels = async () => {
      try {
        const data = await getAllModels()
        availableModels.value = data.models.map(m => m.model_type)
      } catch (error) {
        console.error('加载模型列表失败:', error)
      }
    }

    const prevPage = () => {
      if (currentPage.value > 1) {
        currentPage.value--
        loadHistory()
      }
    }

    const nextPage = () => {
      if (currentPage.value < totalPages.value) {
        currentPage.value++
        loadHistory()
      }
    }

    onMounted(() => {
      loadHistory()
      loadStats()
      loadAvailableModels()
    })

    return {
      loading,
      history,
      stats,
      message,
      messageType,
      currentPage,
      totalPages,
      filters,
      availableModels,
      loadHistory,
      prevPage,
      nextPage
    }
  }
}
</script>

