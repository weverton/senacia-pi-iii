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

    <!-- ========================================= -->
    <!-- MAIN CONTENT                              -->
    <!-- ========================================= -->
    <main class="main-content">
      <div class="video-card">
        <!-- Card Header with Status -->
        <div class="video-header">
          <h2>Live Ponto</h2>
          <div class="status-indicator" :class="{ 'is-online': !hasError }">
            <span class="status-dot"></span>
            <span class="status-text">{{ hasError ? 'Offline' : 'Online' }}</span>
          </div>
        </div>
        
        <!-- Video Area -->
        <div class="video-wrapper">
          <!-- The :key="streamKey" forces Vue to reload the image when we click Retry -->
          <img 
            :key="streamKey"
            :src="streamUrl" 
            alt="IP Camera Stream" 
            class="camera-feed"
            @error="handleStreamError"
            @load="handleStreamLoad"
          />
          
          <!-- Error Overlay -->
          <div v-if="hasError" class="error-overlay">
            <p>Unable to load video feed.</p>
            <button @click="refreshStream" class="btn-retry">Retry Connection</button>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref } from 'vue';

// Point this to your FastAPI backend endpoint
const streamUrl = ref('http://localhost:8000/video_feed');
const hasError = ref(false);
const streamKey = ref(0);

// Triggered if the <img> tag fails to load the stream
const handleStreamError = () => {
  hasError.value = true;
};

// Triggered when the stream successfully loads
const handleStreamLoad = () => {
  hasError.value = false;
};

// Forces Vue to destroy and recreate the <img> tag, triggering a fresh connection
const refreshStream = () => {
  hasError.value = false;
  streamKey.value += 1; 
};
</script>

<style scoped>
/* --- LAYOUT & BACKGROUND --- */
.app-container {
  min-height: 100vh;
  /* Light blue gradient background */
  background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%); 
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px;
}

/* --- HEADER & LOGO --- */
.header {
  text-align: center;
  margin-bottom: 30px;
}

.logo-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 15px;
}

.logo-image {
  width: 120px;
  height: 120px;
  border-radius: 50%; /* Makes the image a perfect circle */
  object-fit: cover;
  border: 4px solid #2563EB; /* Royal Blue border */
  box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
  background-color: #fff;
}

.app-title {
  color: #1E3A8A; /* Deep Blue */
  margin: 0;
  font-size: 2rem;
  font-weight: 700;
  letter-spacing: -0.5px;
}

/* --- MAIN CARD --- */
.main-content {
  width: 100%;
  max-width: 900px;
  display: flex;
  justify-content: center;
}

.video-card {
  background: #ffffff;
  border-radius: 16px;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
  overflow: hidden;
  width: 100%;
}

.video-header {
  background: #1E3A8A; /* Deep Blue header */
  color: white;
  padding: 15px 25px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.video-header h2 {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 600;
}

/* --- STATUS INDICATOR --- */
.status-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.9rem;
  font-weight: 600;
  color: #94A3B8; /* Muted gray for offline */
}

.status-indicator.is-online {
  color: #F97316; /* Vibrant Orange for online */
}

.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background-color: currentColor;
  box-shadow: 0 0 8px currentColor;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0% { opacity: 1; }
  50% { opacity: 0.4; }
  100% { opacity: 1; }
}

/* --- VIDEO AREA --- */
.video-wrapper {
  position: relative;
  background-color: #0F172A; /* Dark slate background for the video */
  width: 100%;
  aspect-ratio: 16 / 9; /* Maintains a 16:9 shape even if stream is loading */
  display: flex;
  justify-content: center;
  align-items: center;
}

.camera-feed {
  width: 100%;
  height: 100%;
  object-fit: contain; /* Ensures the video doesn't get cropped */
  display: block;
}

/* --- ERROR OVERLAY & BUTTON --- */
.error-overlay {
  position: absolute;
  inset: 0; /* Covers the whole video-wrapper */
  background: rgba(15, 23, 42, 0.85);
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  color: white;
  gap: 15px;
  backdrop-filter: blur(4px);
}

.error-overlay p {
  margin: 0;
  font-size: 1.1rem;
  font-weight: 500;
}

.btn-retry {
  background-color: #F97316; /* Vibrant Orange button */
  color: white;
  border: none;
  padding: 12px 28px;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  box-shadow: 0 4px 6px rgba(249, 115, 22, 0.3);
}

.btn-retry:hover {
  background-color: #EA580C; /* Darker orange on hover */
  transform: translateY(-2px);
  box-shadow: 0 6px 10px rgba(249, 115, 22, 0.4);
}

.btn-retry:active {
  transform: translateY(0);
}
</style>