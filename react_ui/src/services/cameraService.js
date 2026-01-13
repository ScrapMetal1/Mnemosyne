// Camera service for communicating with the Flask backend

const API_BASE_URL = 'http://localhost:5000/api';

const formatMetrics = (metrics) => {
  if (!metrics) return '';
  return Object.entries(metrics)
    .map(([key, value]) =>
      typeof value === 'number' ? `${key}=${value.toFixed(1)}ms` : `${key}=${value}`
    )
    .join(', ');
};

const logLatency = (label, start, metrics) => {
  const elapsed = performance.now() - start;
  const stageText = formatMetrics(metrics);
  const suffix = stageText ? ` | stages: ${stageText}` : '';
  console.log(`[latency] ${label}: total ${elapsed.toFixed(1)} ms${suffix}`);
};

export const cameraService = {
  /**
   * Start the camera
   */
  async startCamera() {
    try {
      const start = performance.now();
      const response = await fetch(`${API_BASE_URL}/camera/start`, {
        method: 'POST',
      });
      const data = await response.json();
      logLatency('camera/start', start, data.metrics);
      return data;
    } catch (error) {
      console.error('Error starting camera:', error);
      throw error;
    }
  },

  /**
   * Stop the camera
   */
  async stopCamera() {
    try {
      const start = performance.now();
      const response = await fetch(`${API_BASE_URL}/camera/stop`, {
        method: 'POST',
      });
      const data = await response.json();
      logLatency('camera/stop', start, data.metrics);
      return data;
    } catch (error) {
      console.error('Error stopping camera:', error);
      throw error;
    }
  },

  /**
   * Get camera status
   */
  async getCameraStatus() {
    try {
      const start = performance.now();
      const response = await fetch(`${API_BASE_URL}/camera/status`);
      const data = await response.json();
      logLatency('camera/status', start, data.metrics);
      return data;
    } catch (error) {
      console.error('Error getting camera status:', error);
      throw error;
    }
  },

  /**
   * Get current frame URL
   */
  getFrameUrl() {
    return `${API_BASE_URL}/camera/frame?t=${Date.now()}`;
  },

  /**
   * Analyze the scene with Vision API
   */
  async analyzeScene(options = {}) {
    try {
      const start = performance.now();
      const response = await fetch(`${API_BASE_URL}/scene/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          speak: Boolean(options.speak),
        }),
      });
      const data = await response.json();
      logLatency('scene/analyze', start, data.metrics);
      return data;
    } catch (error) {
      console.error('Error analyzing scene:', error);
      throw error;
    }
  },
};
