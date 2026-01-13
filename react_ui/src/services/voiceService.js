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

export const voiceService = {
  /**
   * Start voice recording
   */
  async startVoice(mode = 'plain') {
    try {
      const start = performance.now();
      const response = await fetch(`${API_BASE_URL}/voice/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ mode }),
      });
      const data = await response.json();
      logLatency(`voice/start:${mode}`, start, data.metrics);
      return data;
    } catch (error) {
      console.error('Error starting voice recording:', error);
      throw error;
    }
  },

  /**
   * Stop voice recording
   */
  async stopVoice() {
    try {
      const start = performance.now();
      const response = await fetch(`${API_BASE_URL}/voice/stop`, {
        method: 'POST',
      });
      const data = await response.json();
      logLatency('voice/stop', start, data.metrics);
      return data;
    } catch (error) {
      console.error('Error stopping voice recording:', error);
      throw error;
    }
  },

  /**
   * Get the latest AI response
   */
  async getVoiceResponse() {
    try {
      const start = performance.now();
      const response = await fetch(`${API_BASE_URL}/voice/response`);
      const data = await response.json();
      logLatency('voice/response', start);
      return data;
    } catch (error) {
      console.error('Error getting voice response:', error);
      throw error;
    }
  },

  /**
   * Stop any TTS playback
   */
  async stopSpeech() {
    try {
      const start = performance.now();
      const response = await fetch(`${API_BASE_URL}/voice/stop-speech`, {
        method: 'POST',
      });
      const data = await response.json();
      logLatency('voice/stop-speech', start, data.metrics);
      return data;
    } catch (error) {
      console.error('Error stopping speech playback:', error);
      throw error;
    }
  },

  /**
   * Get recording/speaking status
   */
  async getVoiceStatus() {
    try {
      const start = performance.now();
      const response = await fetch(`${API_BASE_URL}/voice/status`);
      const data = await response.json();
      logLatency('voice/status', start, data.metrics);
      return data;
    } catch (error) {
      console.error('Error getting voice status:', error);
      throw error;
    }
  },
};