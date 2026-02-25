// Voice service for managing local MediaRecorder and communicating with Flask

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

let mediaRecorder = null;
let audioChunks = [];
let localStream = null;
let currentMode = 'plain';

export const voiceService = {
  /**
   * Start local voice recording
   */
  async startVoice(mode = 'plain') {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      localStream = stream;
      audioChunks = [];
      
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };
      
      mediaRecorder.start();
      currentMode = mode;
      
      return { status: 'success' };
    } catch (error) {
      console.error('Error starting voice recording:', error);
      return { status: 'error', message: error.message || error.name };
    }
  },

  /**
   * Stop local voice recording and send to backend
   */
  async stopVoice(imageBase64 = null, onEvent) {
    if (!mediaRecorder || mediaRecorder.state === 'inactive') {
      return { status: 'error', message: 'Not recording' };
    }

    return new Promise((resolve, reject) => {
      mediaRecorder.onstop = async () => {
        try {
          // Detect the actual mime type the browser recorded with
          const mimeType = mediaRecorder.mimeType || 'audio/webm';
          const ext = mimeType.includes('mp4') ? 'mp4' : 'webm';
          const audioBlob = new Blob(audioChunks, { type: mimeType });
          
          if (localStream) {
            localStream.getTracks().forEach(track => track.stop());
            localStream = null;
          }

          const formData = new FormData();
          formData.append('audio', audioBlob, `recording.${ext}`);
          formData.append('mode', currentMode);
          
          if (currentMode === 'scene' && imageBase64) {
            formData.append('image', imageBase64);
          }

          const start = performance.now();
          const response = await fetch(`${API_BASE_URL}/voice/process`, {
            method: 'POST',
            body: formData,
          });

          if (!response.ok || !response.body) {
             throw new Error(`Server returned ${response.status}`);
          }
          
          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer = '';

          let finalResult = { status: 'success', data: { mode: currentMode }, metrics: {} };

          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // keep the last incomplete chunk in buffer
            
            for (const line of lines) {
              if (line.startsWith('data: ')) {
                try {
                  const data = JSON.parse(line.slice(6));
                  if (onEvent) onEvent(data);

                  if (data.type === 'transcript') {
                     finalResult.data.transcript = data.text;
                  } else if (data.type === 'done') {
                     finalResult.metrics = data.metrics;
                     finalResult.data.response = data.full_response;
                  }
                } catch (e) {
                  console.error("Error parsing SSE JSON:", e);
                }
              }
            }
          }

          logLatency('voice/process_stream', start, finalResult.metrics);
          resolve(finalResult);
        } catch (error) {
          console.error('Error processing voice:', error);
          reject(error);
        }
      };

      mediaRecorder.stop();
    });
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