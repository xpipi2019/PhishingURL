<template>
  <div>
    <div class="card">
      <h2>系统健康检查</h2>
      <button class="btn btn-secondary" @click="checkHealth" :disabled="loading">
        {{ loading ? '检查中...' : '刷新检查' }}
      </button>
    </div>

    <div v-if="health" class="card">
      <h2>系统状态</h2>
      
      <div class="form-group">
        <strong>整体状态:</strong> 
        <span :class="['badge', health.status === 'healthy' ? 'badge-success' : 'badge-danger']">
          {{ health.status === 'healthy' ? '健康' : health.status === 'degraded' ? '降级' : '异常' }}
        </span>
      </div>

      <div class="form-group">
        <strong>已训练模型数:</strong> {{ health.model_count }}
      </div>

      <div class="form-group" v-if="health.best_model">
        <strong>最佳模型:</strong> 
        <span class="badge badge-success">{{ health.best_model }}</span>
        <span style="margin-left: 1rem;">准确率: {{ (health.best_accuracy * 100).toFixed(2) }}%</span>
      </div>

      <div class="form-group">
        <strong>可以预测:</strong> 
        <span :class="['badge', health.can_predict ? 'badge-success' : 'badge-danger']">
          {{ health.can_predict ? '是' : '否' }}
        </span>
      </div>

      <div class="form-group">
        <strong>模型工作正常:</strong> 
        <span :class="['badge', health.model_working ? 'badge-success' : 'badge-danger']">
          {{ health.model_working ? '是' : '否' }}
        </span>
      </div>

      <div class="form-group">
        <strong>运行时长:</strong> {{ health.uptime_formatted }}
      </div>
    </div>

    <div v-if="health && health.system_health" class="card">
      <h2>系统健康详情</h2>
      
      <div class="stats-grid">
        <div class="stat-card">
          <h3 :style="{ color: health.database_connected ? '#27ae60' : '#e74c3c' }">
            {{ health.database_connected ? '✓' : '✗' }}
          </h3>
          <p>数据库连接</p>
          <div v-if="health.system_health.checks.database">
            <small>{{ health.system_health.checks.database.status }}</small>
          </div>
        </div>

        <div class="stat-card">
          <h3 :style="{ color: health.disk_healthy ? '#27ae60' : '#e74c3c' }">
            {{ health.disk_healthy ? '✓' : '✗' }}
          </h3>
          <p>磁盘空间</p>
          <div v-if="health.system_health.checks.disk">
            <small>
              可用: {{ health.system_health.checks.disk.free_gb?.toFixed(2) }} GB<br>
              使用率: {{ health.system_health.checks.disk.used_percent?.toFixed(1) }}%
            </small>
          </div>
        </div>

        <div class="stat-card">
          <h3 :style="{ color: health.memory_healthy ? '#27ae60' : '#e74c3c' }">
            {{ health.memory_healthy ? '✓' : '✗' }}
          </h3>
          <p>内存使用</p>
          <div v-if="health.system_health.checks.memory">
            <small>
              可用: {{ health.system_health.checks.memory.available_mb?.toFixed(2) }} MB<br>
              使用率: {{ health.system_health.checks.memory.used_percent?.toFixed(1) }}%
            </small>
          </div>
        </div>

        <div class="stat-card">
          <h3 :style="{ color: health.cpu_healthy ? '#27ae60' : '#e74c3c' }">
            {{ health.cpu_healthy ? '✓' : '✗' }}
          </h3>
          <p>CPU使用率</p>
          <div v-if="health.system_health.checks.cpu">
            <small>{{ health.system_health.checks.cpu.cpu_percent?.toFixed(1) }}%</small>
          </div>
        </div>
      </div>

      <div class="form-group" style="margin-top: 1rem;">
        <strong>所有检查通过:</strong> 
        <span :class="['badge', health.system_health.all_healthy ? 'badge-success' : 'badge-danger']">
          {{ health.system_health.all_healthy ? '是' : '否' }}
        </span>
      </div>
    </div>

    <!-- 消息提示 -->
    <div v-if="message" :class="['alert', messageType]">
      {{ message }}
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'
import { healthCheck } from '../api/services'

export default {
  name: 'HealthView',
  setup() {
    const loading = ref(false)
    const health = ref(null)
    const message = ref('')
    const messageType = ref('')

    const showMessage = (msg, type = 'info') => {
      message.value = msg
      messageType.value = `alert-${type}`
      setTimeout(() => {
        message.value = ''
      }, 5000)
    }

    const checkHealth = async () => {
      loading.value = true
      try {
        health.value = await healthCheck()
      } catch (error) {
        showMessage(error.message || '健康检查失败', 'error')
      } finally {
        loading.value = false
      }
    }

    onMounted(() => {
      checkHealth()
      // 每30秒自动刷新
      setInterval(() => {
        checkHealth()
      }, 30000)
    })

    return {
      loading,
      health,
      message,
      messageType,
      checkHealth
    }
  }
}
</script>

