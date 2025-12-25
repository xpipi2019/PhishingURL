import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'
import './style.css'

import TrainView from './views/TrainView.vue'
import PredictView from './views/PredictView.vue'
import ModelsView from './views/ModelsView.vue'
import HistoryView from './views/HistoryView.vue'
import HealthView from './views/HealthView.vue'

const routes = [
  { path: '/', redirect: '/predict' },
  { path: '/train', component: TrainView },
  { path: '/predict', component: PredictView },
  { path: '/models', component: ModelsView },
  { path: '/history', component: HistoryView },
  { path: '/health', component: HealthView }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

const app = createApp(App)
app.use(router)
app.mount('#app')

