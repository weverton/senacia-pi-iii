<template>
  <div class="app-container">
    <!-- ========================================= -->
    <!-- HEADER / LOGO SPACE                       -->
    <!-- ========================================= -->
    <header class="header">
      <div class="logo-container">
        <!-- 
          INSTRUCTIONS: Replace the URL below with your own image.
          If using a local file, place it in the 'src/assets' folder and import it, 
          or just put it in the 'public' folder and use '/your-image.png'.
        -->
        <img 
          src="/images/senac-logo.png" 
          alt="Company Logo" 
          class="logo-image" 
        />
        <h1 class="app-title">Ponto Monitor</h1>
      </div>
    </header>

    <!-- MAIN CONTENT -->
    <main class="main-content">
      <!-- VIDEO CARD -->
      <div class="video-card">
        <div class="video-header">
          <h2>Live Monitor</h2>
          <div class="status-indicator" :class="{ 'is-online': !hasError }">
            <span class="status-dot"></span>
            <span class="status-text">{{ hasError ? 'Offline' : 'Online' }}</span>
          </div>
        </div>
        
        <div class="video-wrapper">
          <img 
            :key="streamKey"
            :src="streamUrl" 
            alt="IP Camera Stream" 
            class="camera-feed"
            @error="handleStreamError"
            @load="handleStreamLoad"
          />
          
          <div v-if="hasError" class="error-overlay">
            <p>Unable to load video feed.</p>
            <button @click="refreshStream" class="btn-retry">Retry Connection</button>
          </div>
        </div>
      </div>

      <!-- RECOGNITION LOGS CARD -->
      <div class="logs-card">
        <div class="logs-header">
          <h2>Log Reconhecimentos</h2>
          <div class="logs-status">
            <span class="status-dot" :class="{ active: wsConnected }"></span>
            <span>{{ wsConnected ? 'Listening' : 'Disconnected' }}</span>
          </div>
        </div>

        <div class="logs-container" ref="logsContainerRef">
          <!-- Empty state -->
          <div v-if="logs.length === 0" class="empty-state">
            <div class="empty-icon">👤</div>
            <p>Nenhum reconhecimento.</p>
            <p class="empty-subtitle">
              Reconhecimentos vão aparecer aqui (5 min limite por pessoa).
            </p>
          </div>

          <!-- Log entries -->
          <div 
            v-for="log in logs" 
            :key="log.timestamp" 
            class="log-entry"
          >
            <!-- Face thumbnail -->
            <div class="log-face">
              <img 
                v-if="log.face_image" 
                :src="log.face_image" 
                :alt="log.person_name"
                class="face-thumb"
              />
              <div v-else class="face-placeholder">?</div>
            </div>

            <!-- Info -->
            <div class="log-info">
              <div class="log-name">{{ log.person_name }}</div>
              <div class="log-meta">
                <span class="log-confidence">
                  {{ (log.confidence * 100).toFixed(1) }}% confiança
                </span>
                <span class="log-time">{{ formatTime(log.formatted_time) }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, nextTick } from 'vue';

// --- Video Stream State ---
const streamUrl = ref('http://localhost:8000/video_feed');
const hasError = ref(false);
const streamKey = ref(0);

const handleStreamError = () => { hasError.value = true; };
const handleStreamLoad = () => { hasError.value = false; };
const refreshStream = () => {
  hasError.value = false;
  streamKey.value += 1;
};

// --- WebSocket Logs State ---
const logs = ref([]);
const wsConnected = ref(false);
const logsContainerRef = ref(null);
let websocket = null;
let reconnectTimer = null;

const WS_URL = 'ws://localhost:8000/ws/logs';

const connectWebSocket = () => {
  if (websocket && websocket.readyState === WebSocket.OPEN) return;

  console.log('🔌 Attempting to connect to:', WS_URL);
  websocket = new WebSocket(WS_URL);

  websocket.onopen = () => {
    console.log('✅ Connected to logs WebSocket');
    wsConnected.value = true;
  };

  websocket.onmessage = (event) => {
    console.log('📨 Received message:', event.data);
    try {
      const logEntry = JSON.parse(event.data);
      logs.value.unshift(logEntry);
      
      if (logs.value.length > 50) {
        logs.value = logs.value.slice(0, 50);
      }
      
      nextTick(() => {
        if (logsContainerRef.value) {
          logsContainerRef.value.scrollTop = 0;
        }
      });
    } catch (e) {
      console.error('❌ Failed to parse log entry:', e);
    }
  };

  websocket.onerror = (error) => {
    console.error('❌ WebSocket error:', error);
  };

  websocket.onclose = (event) => {
    console.log('🔌 WebSocket closed. Code:', event.code, 'Reason:', event.reason);
    wsConnected.value = false;
    reconnectTimer = setTimeout(connectWebSocket, 3000);
  };
};

// Format ISO timestamp to readable time
// Format ISO timestamp to readable time (now handles timezone correctly)
const formatTime = (isoString) => {
  if (!isoString) return '';
  const date = new Date(isoString);
  
  // JavaScript automatically converts timezone-aware ISO strings to local time
  return date.toLocaleString('pt-BR', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false
  });
};

onMounted(() => {
  connectWebSocket();
});

onUnmounted(() => {
  if (websocket) websocket.close();
  if (reconnectTimer) clearTimeout(reconnectTimer);
});
</script>

<style scoped>
.app-container {
  min-height: 100vh;
  background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px;
}

/* HEADER */
.header { text-align: center; margin-bottom: 30px; }
.logo-container { display: flex; flex-direction: column; align-items: center; gap: 15px; }
.logo-image {
  width: 120px; height: 120px; border-radius: 50%;
  object-fit: cover; border: 4px solid #2563EB;
  box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
  background-color: #fff;
}
.app-title { color: #1E3A8A; margin: 0; font-size: 2rem; font-weight: 700; }

/* MAIN LAYOUT */
.main-content {
  width: 100%; max-width: 1200px;
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 20px;
}

@media (max-width: 900px) {
  .main-content { grid-template-columns: 1fr; }
}

/* VIDEO CARD */
.video-card {
  background: #ffffff; border-radius: 16px;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
  overflow: hidden;
}
.video-header {
  background: #1E3A8A; color: white;
  padding: 15px 25px;
  display: flex; justify-content: space-between; align-items: center;
}
.video-header h2 { margin: 0; font-size: 1.25rem; font-weight: 600; }

.status-indicator {
  display: flex; align-items: center; gap: 8px;
  font-size: 0.9rem; font-weight: 600; color: #94A3B8;
}
.status-indicator.is-online { color: #F97316; }

.status-dot {
  width: 10px; height: 10px; border-radius: 50%;
  background-color: currentColor;
  box-shadow: 0 0 8px currentColor;
}
.status-indicator.is-online .status-dot { animation: pulse 2s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

.video-wrapper {
  position: relative; background-color: #0F172A;
  width: 100%; aspect-ratio: 16 / 9;
  display: flex; justify-content: center; align-items: center;
}
.camera-feed { width: 100%; height: 100%; object-fit: contain; display: block; }

.error-overlay {
  position: absolute; inset: 0;
  background: rgba(15, 23, 42, 0.85);
  display: flex; flex-direction: column;
  justify-content: center; align-items: center;
  color: white; gap: 15px; backdrop-filter: blur(4px);
}
.btn-retry {
  background-color: #F97316; color: white; border: none;
  padding: 12px 28px; border-radius: 8px;
  font-size: 1rem; font-weight: 600; cursor: pointer;
  transition: all 0.2s ease;
  box-shadow: 0 4px 6px rgba(249, 115, 22, 0.3);
}
.btn-retry:hover { background-color: #EA580C; transform: translateY(-2px); }

/* LOGS CARD */
.logs-card {
  background: #ffffff; border-radius: 16px;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
  overflow: hidden;
  display: flex; flex-direction: column;
  max-height: 600px;
}
.logs-header {
  background: #1E3A8A; color: white;
  padding: 15px 20px;
  display: flex; justify-content: space-between; align-items: center;
}
.logs-header h2 { margin: 0; font-size: 1.1rem; font-weight: 600; }
.logs-status {
  display: flex; align-items: center; gap: 8px;
  font-size: 0.85rem; color: #94A3B8;
}
.logs-status .status-dot {
  background-color: #94A3B8;
  box-shadow: 0 0 6px #94A3B8;
}
.logs-status .status-dot.active {
  background-color: #F97316;
  box-shadow: 0 0 8px #F97316;
  animation: pulse 2s infinite;
}

.logs-container {
  flex: 1; overflow-y: auto;
  padding: 10px;
  background: #F8FAFC;
}

/* Empty state */
.empty-state {
  text-align: center; padding: 40px 20px; color: #94A3B8;
}
.empty-icon { font-size: 3rem; margin-bottom: 10px; }
.empty-state p { margin: 5px 0; }
.empty-subtitle { font-size: 0.85rem; color: #CBD5E1; }

/* Log entry */
.log-entry {
  display: flex; align-items: center; gap: 12px;
  padding: 12px;
  background: white;
  border-radius: 10px;
  margin-bottom: 8px;
  border-left: 4px solid #F97316;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
  transition: transform 0.2s;
  animation: slideIn 0.3s ease-out;
}
.log-entry:hover { transform: translateX(2px); }

@keyframes slideIn {
  from { opacity: 0; transform: translateY(-10px); }
  to { opacity: 1; transform: translateY(0); }
}

.log-face {
  flex-shrink: 0;
  width: 56px; height: 56px;
  border-radius: 50%;
  overflow: hidden;
  background: #DBEAFE;
  display: flex; align-items: center; justify-content: center;
  border: 2px solid #2563EB;
}
.face-thumb { width: 100%; height: 100%; object-fit: cover; }
.face-placeholder { color: #2563EB; font-size: 1.5rem; font-weight: bold; }

.log-info { flex: 1; min-width: 0; }
.log-name {
  font-weight: 700; color: #1E3A8A;
  font-size: 1rem;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.log-meta {
  display: flex; justify-content: space-between;
  font-size: 0.8rem; color: #64748B;
  margin-top: 4px;
}
.log-confidence { color: #F97316; font-weight: 600; }
.log-time { color: #94A3B8; }

/* Custom scrollbar */
.logs-container::-webkit-scrollbar { width: 6px; }
.logs-container::-webkit-scrollbar-track { background: transparent; }
.logs-container::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }
.logs-container::-webkit-scrollbar-thumb:hover { background: #94A3B8; }
</style>