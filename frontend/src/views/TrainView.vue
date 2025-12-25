<template>
  <div>
    <div class="card">
      <h2>模型训练</h2>
      
      <div class="form-group">
        <label>数据集路径</label>
        <input 
          v-model="form.dataset_path" 
          type="text" 
          placeholder="data/raw/PhishingData.csv"
        />
      </div>

      <div class="form-row">
        <div class="form-group">
          <label>模型类型</label>
          <select v-model="form.model_type">
            <option value="logistic_regression">Logistic Regression</option>
            <option value="knn">KNN</option>
            <option value="svm">SVM</option>
            <option value="kernel_svm">Kernel SVM</option>
            <option value="naive_bayes">Naive Bayes</option>
            <option value="decision_tree">Decision Tree</option>
            <option value="random_forest">Random Forest</option>
            <option value="xgboost">XGBoost</option>
          </select>
        </div>

        <div class="form-group">
          <label>测试集比例</label>
          <input 
            v-model.number="form.test_size" 
            type="number" 
            step="0.05" 
            min="0.1" 
            max="0.5"
          />
        </div>
      </div>

      <div class="form-group">
        <label>随机种子</label>
        <input 
          v-model.number="form.random_state" 
          type="number"
        />
      </div>

      <button 
        class="btn btn-primary" 
        @click="handleTrain" 
        :disabled="loading || !form.dataset_path || !form.model_type"
      >
        {{ loading ? '训练中...' : '开始训练' }}
      </button>

      <!-- 消息提示 -->
      <div v-if="message" :class="['alert', messageType]">
        {{ message }}
      </div>
    </div>

    <!-- 训练结果 -->
    <div v-if="trainResult" class="card">
      <h2>训练结果</h2>
      <div class="form-group">
        <strong>模型类型:</strong> {{ trainResult.model_type }}
      </div>
      <div class="form-group">
        <strong>准确率:</strong> 
        <span style="font-size: 1.2rem; color: #27ae60; font-weight: bold;">
          {{ (trainResult.accuracy * 100).toFixed(2) }}%
        </span>
        <span v-if="trainResult.is_best_model" class="badge badge-success" style="margin-left: 1rem;">
          最佳模型
        </span>
      </div>
      <div class="form-group">
        <strong>训练时间:</strong> {{ trainResult.training_time.toFixed(2) }} 秒
      </div>
      <div class="form-group">
        <strong>混淆矩阵:</strong>
        <pre style="background: #f5f5f5; padding: 1rem; border-radius: 4px; margin-top: 0.5rem;">
{{ JSON.stringify(trainResult.confusion_matrix, null, 2) }}
        </pre>
      </div>
      <div class="form-group" v-if="trainResult.data_cleaning">
        <strong>数据清理:</strong>
        <ul style="margin-top: 0.5rem; padding-left: 1.5rem;">
          <li>原始样本数: {{ trainResult.data_cleaning.original_samples }}</li>
          <li>清理后样本数: {{ trainResult.data_cleaning.cleaned_samples }}</li>
          <li>移除样本数: {{ trainResult.data_cleaning.removed_samples }}</li>
        </ul>
      </div>
      <div class="form-group" v-if="trainResult.classification_report">
        <strong>分类报告:</strong>
        <pre style="background: #f5f5f5; padding: 1rem; border-radius: 4px; margin-top: 0.5rem; max-height: 300px; overflow-y: auto;">
{{ JSON.stringify(trainResult.classification_report, null, 2) }}
        </pre>
      </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import { trainModel } from '../api/services'

export default {
  name: 'TrainView',
  setup() {
    const form = ref({
      dataset_path: 'data/raw/PhishingData.csv',
      model_type: 'logistic_regression',
      test_size: 0.25,
      random_state: 0
    })
    const loading = ref(false)
    const message = ref('')
    const messageType = ref('')
    const trainResult = ref(null)

    const showMessage = (msg, type = 'info') => {
      message.value = msg
      messageType.value = `alert-${type}`
      setTimeout(() => {
        message.value = ''
      }, 10000)
    }

    const handleTrain = async () => {
      loading.value = true
      trainResult.value = null
      message.value = ''

      try {
        const data = await trainModel(form.value)
        trainResult.value = data
        showMessage(data.message || '训练完成', 'success')
      } catch (error) {
        showMessage(error.message || '训练失败', 'error')
      } finally {
        loading.value = false
      }
    }

    return {
      form,
      loading,
      message,
      messageType,
      trainResult,
      handleTrain
    }
  }
}
</script>

