// Camera service for communicating with the Flask backend and managing local camera

const API_BASE_URL = '/api';

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
   * Start the local camera
   */
  async startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "environment", // Prefer back camera on mobile
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false // Audio is handled separately
      });
      return { status: 'success', stream };
    } catch (error) {
      console.error('Error starting local camera:', error);
      return { status: 'error', message: error.message || error.name };
    }
  },

  /**
   * Stop the local camera stream
   */
  stopCamera(stream) {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
    }
    return { status: 'success' };
  },

  /**
   * Analyze the scene with Vision API using a captured frame
   */
  async analyzeScene(imageBase64, options = {}) {
    try {
      const start = performance.now();
      const response = await fetch(`${API_BASE_URL}/scene/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageBase64,
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
