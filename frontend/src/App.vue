<template>
  <div id="app">
    <nav class="navbar">
      <div class="container">
        <h1>网络安全威胁检测系统</h1>
        <nav>
          <router-link to="/predict">URL预测</router-link>
          <router-link to="/train">模型训练</router-link>
          <router-link to="/models">模型管理</router-link>
          <router-link to="/history">预测历史</router-link>
          <router-link to="/health">健康检查</router-link>
        </nav>
      </div>
    </nav>
    <main class="container">
      <router-view />
    </main>
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>

