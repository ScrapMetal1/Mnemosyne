import './App.css'
import Orb from './components/orb'
import { useState, useEffect, useRef } from 'react'
import { CiCamera, CiMicrophoneOn } from 'react-icons/ci'
import { FiImage, FiMic, FiMicOff, FiStopCircle, FiActivity, FiAperture  } from 'react-icons/fi'
import { cameraService } from './services/cameraService'
import { voiceService } from './services/voiceService'
import { memoryService} from './services/memoryService'

function App() {
  const [isCameraRunning, setIsCameraRunning] = useState(false)
  const [cameraStream, setCameraStream] = useState(null)
  const videoRef = useRef(null)

  const [sceneAnalysis, setSceneAnalysis] = useState('')
  const [analysisLoading, setAnalysisLoading] = useState(false)
  const [voiceResult, setVoiceResult] = useState(null)
  const [isRecording, setIsRecording] = useState(false)
  const [recordingMode, setRecordingMode] = useState(null)
  const [isSpeechActive, setIsSpeechActive] = useState(false)
  const [uiMessage, setUiMessage] = useState(null)
  const [latencyMetrics, setLatencyMetrics] = useState(null)
  const [memoryLoading, setMemoryLoading] = useState(false)

  // Attach stream to video element when it changes
  useEffect(() => {
    if (videoRef.current && cameraStream) {
      videoRef.current.srcObject = cameraStream;
      videoRef.current.play().catch(e => console.error("Video play error:", e));
    }
  }, [cameraStream]);

  // Clean up camera on unmount
  useEffect(() => {
    return () => {
      if (cameraStream) {
        cameraService.stopCamera(cameraStream);
      }
    };
  }, [cameraStream]);

  useEffect(() => {
    if (!isRecording && !isSpeechActive) return
    const interval = setInterval(async () => {
      try {
        const status = await voiceService.getVoiceStatus()
        if (typeof status.is_speaking === 'boolean') {
          // Commented out to prevent conflict with local speech active state
          // setIsSpeechActive(status.is_speaking)
        }
        if (typeof status.is_recording === 'boolean' && !status.is_recording) {
          // setIsRecording(false)
          // setRecordingMode(null)
        }
      } catch (error) {
        console.error('Error fetching voice status:', error)
      }
    }, 1200)
    return () => clearInterval(interval)
  }, [isRecording, isSpeechActive])

  const handleCameraToggle = async () => {
    try {
      if (isCameraRunning) {
        cameraService.stopCamera(cameraStream)
        setCameraStream(null)
        setIsCameraRunning(false)
      } else {
        const result = await cameraService.startCamera()
        if (result.status === 'success') {
          setCameraStream(result.stream)
          setIsCameraRunning(true)
        } else {
          setUiMessage(result.message || 'Unable to start camera')
        }
      }
    } catch (error) {
      console.error('Error toggling camera:', error)
      setUiMessage(`Camera error: ${error.message || error.name || 'Unknown error'}`)
    }
  }

  // Helper to capture a frame from the video element
  const captureFrame = () => {
    if (!videoRef.current || !isCameraRunning) {
      console.warn("captureFrame failed: videoRef or isCameraRunning is falsy");
      return null;
    }
    const width = videoRef.current.videoWidth;
    const height = videoRef.current.videoHeight;
    
    if (!width || !height) {
      console.warn("captureFrame failed: video dimensions are zero", { width, height });
      return null;
    }

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0, width, height);
    // Get base64 string, remove the data:image/jpeg;base64, prefix for the backend
    const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
    return dataUrl.split(',')[1];
  }

  const handleAnalyze = async (shouldSpeak = false) => {
    setAnalysisLoading(true)
    try {
      const imageBase64 = captureFrame();
      if (!imageBase64) {
        throw new Error("No camera frame available");
      }
      const result = await cameraService.analyzeScene(imageBase64, { speak: shouldSpeak })
      setSceneAnalysis(result.analysis)
      if (shouldSpeak) {
        setIsSpeechActive(true)
      }
    } catch (error) {
      console.error('Error analyzing scene:', error)
      setUiMessage('Could not analyze scene. Check backend logs.')
    } finally {
      setAnalysisLoading(false)
    }
  }

  const handleVoiceStart = async (mode) => {
    try {
      if (isRecording && recordingMode === mode) {
        await handleVoiceStop()
        return
      }

      if (isRecording && recordingMode !== mode) {
        await handleVoiceStop()
      }

      const result = await voiceService.startVoice(mode)
      if (result.status === 'success') {
        setIsRecording(true)
        setRecordingMode(mode)
        setLatencyMetrics(null) // Clear previous metrics
        setUiMessage(`Listening (${mode === 'scene' ? 'scene aware' : 'standard'})…`)
      } else {
        setUiMessage(result.message || 'Unable to start recording')
      }
    } catch (error) {
      console.error('Error starting voice recording:', error)
      setUiMessage(`Mic error: ${error.message || error.name || 'Unknown error'}`)
    }
  }

  const handleVoiceStop = async () => {
    try {
      setUiMessage('Processing...')
      let imageBase64 = null;
      if (recordingMode === 'scene') {
        imageBase64 = captureFrame();
      }
      
      setIsRecording(false)
      setRecordingMode(null)
      
      let currentResponse = '';
      setVoiceResult({ transcript: '...', response: '', mode: recordingMode });
      setIsSpeechActive(true);

      let audioQueue = [];
      let isPlaying = false;

      const playNextAudio = () => {
        if (audioQueue.length === 0) {
          isPlaying = false;
          return;
        }
        isPlaying = true;
        const audio = audioQueue.shift();
        audio.onended = playNextAudio;
        audio.play().catch(e => {
          console.error("Audio playback error:", e);
          playNextAudio(); // Skip to next if there's an error
        });
      };

      const result = await voiceService.stopVoice(imageBase64, (event) => {
         if (event.type === 'transcript') {
             setVoiceResult(prev => ({ ...prev, transcript: event.text }));
         } else if (event.type === 'text') {
             currentResponse += event.text + ' ';
             setVoiceResult(prev => ({ ...prev, response: currentResponse }));
         } else if (event.type === 'audio') {
             const audio = new window.Audio('data:audio/mp3;base64,' + event.audio);
             audioQueue.push(audio);
             if (!isPlaying) {
               playNextAudio();
             }
         }
      })
      
      console.log('Voice stop result:', result)
      
      if (result.status === 'success') {
        const checkDoneInterval = setInterval(() => {
          if (!isPlaying && audioQueue.length === 0) {
            clearInterval(checkDoneInterval);
            setIsSpeechActive(false);
          }
        }, 500);

        if (result.metrics) {
          setLatencyMetrics(result.metrics)
        }
        setUiMessage(null)
      } else {
        setUiMessage(result.message || 'No response generated.')
        setIsSpeechActive(false)
      }
    } catch (error) {
      console.error('Error stopping voice recording:', error)
      setUiMessage('Could not process the recording. Please try again.')
      setIsRecording(false)
      setRecordingMode(null)
      setIsSpeechActive(false)
    }
  }

  const handleStopSpeech = async () => {
    try {
      await voiceService.stopSpeech()
      setIsSpeechActive(false)
    } catch (error) {
      console.error('Error stopping speech:', error)
    }
  }

  const handleSaveMemory = async() => {
    setMemoryLoading(true)
    try {
      const imageBase64 = captureFrame();
      if (!imageBase64) {
        throw new Error("No camera frame available");
      }
      const result = await memoryService.captureMemory(imageBase64)
      if (result.status === 'success'){
        setUiMessage(`Memory saved: ${result.description}`)
      } else{
        setUiMessage(`Error: ${result.message}`)
      }
    }catch (err){
      console.error("Save memory error:", err);
      setUiMessage(`Could not save memory: ${err.message || "Unknown error"}`)
    }finally {
      setMemoryLoading(false)
    }
  }

  const formatMs = (ms) => {
    if (ms === undefined || ms === null) return '—'
    return `${Math.round(ms)}ms`
  }

  return (
    <div className="app-root">
      <div className="glass-shell">
        <section className="primary-panel">
          <header className="status-header">
            <div className="orb-wrapper">
              <Orb
                hoverIntensity={0.5}
                rotateOnHover
                hue={0}
                forceHoverState={isCameraRunning || isRecording || isSpeechActive}
              />
            </div>
            <div className="status-text">
              <p className="eyebrow">Mnemosyne</p>
              <h1>Hello User</h1>
              <div className="chip-row">
                <span className={`status-chip ${isCameraRunning ? 'on' : ''}`}>
                  <CiCamera /> {isCameraRunning ? 'Camera live' : 'Camera idle'}
                </span>
                <span className={`status-chip ${isRecording ? 'on' : ''}`}>
                  <CiMicrophoneOn /> {isRecording ? 'Listening' : 'Standing by'}
                </span>
                <span className={`status-chip ${isSpeechActive ? 'on' : ''}`}>
                  <FiStopCircle /> {isSpeechActive ? 'Speaking' : 'Silent'}
                </span>
              </div>
            </div>
            <div className="header-actions">
              <button
                className={`pill-button primary ${isCameraRunning ? 'active' : ''}`}
                onClick={handleCameraToggle}
              >
                <CiCamera />
                {isCameraRunning ? 'Stop Camera' : 'Start Camera'}
              </button>
              <button
                className={`pill-button outline ${isRecording ? 'active' : ''}`}
                onClick={() => handleVoiceStart('plain')}
              >
                <CiMicrophoneOn />
                {isRecording && recordingMode === 'plain' ? 'Stop Listening' : 'Talk'}
              </button>
            </div>
          </header>

          <div className="controls-grid">
            <div className="card control-card">
              <div className="card-header">
                <span>Scene controls</span>
                <p className="muted">Analyze the camera feed with AI vision.</p>
              </div>
              <div className="control-buttons">
                <button
                  className="pill-button outline"
                  onClick={() => handleAnalyze(false)}
                  disabled={!isCameraRunning || analysisLoading}
                >
                  <FiImage />
                  Describe scene
                </button>
                <button
                  className="pill-button outline"
                  onClick={() => handleAnalyze(true)}
                  disabled={!isCameraRunning || analysisLoading}
                >
                  <FiImage />
                  Describe &amp; speak
                </button>
                <button
                  className="pill-button outline"
                  onClick={handleSaveMemory}
                  disabled= {!isCameraRunning || memoryLoading}
                >
                  <FiAperture />
                  Save Memory
                </button>
              </div>
              {analysisLoading && <p className="hint">Analyzing scene…</p>}
            </div>

            <div className="card control-card">
              <div className="card-header">
                <span>Voice assistant</span>
                <p className="muted">Start a conversation or add scene context.</p>
              </div>
              <div className="control-buttons voice">
                <button
                  className={`pill-button primary ${recordingMode === 'plain' ? 'active' : ''}`}
                  onClick={() => handleVoiceStart('plain')}
                >
                  <FiMic /> Talk
                </button>
                <button
                  className={`pill-button primary ${recordingMode === 'scene' ? 'active' : ''}`}
                  onClick={() => handleVoiceStart('scene')}
                  disabled={!isCameraRunning}
                >
                  <FiImage /> Talk w/ scene
                </button>
                <button
                  className="pill-button outline"
                  onClick={handleVoiceStop}
                  disabled={!isRecording}
                >
                  <FiMicOff /> Stop recording
                </button>
                <button
                  className="pill-button ghost"
                  onClick={handleStopSpeech}
                  disabled={!isSpeechActive}
                >
                  <FiStopCircle /> Stop speech
                </button>
              </div>
            </div>
          </div>

          {latencyMetrics && Object.keys(latencyMetrics).length > 0 && (
            <div className="card metrics-card">
              <div className="card-header">
                <span><FiActivity /> Pipeline Latency</span>
                {latencyMetrics.time_to_first_audio_ms !== undefined && (
                  <span className="metric-highlight">
                    ⚡ {formatMs(latencyMetrics.time_to_first_audio_ms)} to first audio
                  </span>
                )}
              </div>
              <div className="metrics-grid">
                <div className="metric-item">
                  <span className="metric-label">Audio Save</span>
                  <span className="metric-value">{formatMs(latencyMetrics.audio_save_ms)}</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Transcription</span>
                  <span className="metric-value">{formatMs(latencyMetrics.transcription_ms)}</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">LLM First Token</span>
                  <span className="metric-value">{formatMs(latencyMetrics.llm_first_token_ms)}</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">First Sentence</span>
                  <span className="metric-value">{formatMs(latencyMetrics.first_sentence_ms)}</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">First TTS Ready</span>
                  <span className="metric-value">{formatMs(latencyMetrics.first_tts_ready_ms)}</span>
                </div>
                <div className="metric-item primary">
                  <span className="metric-label">Time to First Audio</span>
                  <span className="metric-value">{formatMs(latencyMetrics.time_to_first_audio_ms)}</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Text Generation</span>
                  <span className="metric-value">{formatMs(latencyMetrics.total_text_generation_ms)}</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Total Pipeline</span>
                  <span className="metric-value">{formatMs(latencyMetrics.pipeline_total_ms)}</span>
                </div>
              </div>
            </div>
          )}

          <div className="card analysis-card">
            <div className="card-header">
              <span>AI scene description</span>
            </div>
            <p className={sceneAnalysis ? '' : 'muted'}>
              {sceneAnalysis || 'Request an analysis to receive a concise description here.'}
            </p>
          </div>

          <div className="card voice-card">
            <div className="card-header">
              <span>Voice transcript &amp; response</span>
            </div>
            {voiceResult ? (
              <>
                <p className="label">
                  Heard ({voiceResult.mode === 'scene' ? 'scene aware' : 'standard'}):
                </p>
                <p className="bubble">{voiceResult.transcript}</p>
                <p className="label">Assistant said:</p>
                <p className="bubble">{voiceResult.response}</p>
              </>
            ) : (
              <p className="muted">Ask something with the Talk buttons to see responses here.</p>
            )}
          </div>
        </section>

        <section className="preview-panel">
          <div className="preview-inner">
            <video 
              ref={videoRef} 
              autoPlay 
              playsInline 
              muted 
              onLoadedMetadata={() => videoRef.current && videoRef.current.play()}
              style={{ width: '100%', height: '100%', objectFit: 'cover', visibility: isCameraRunning ? 'visible' : 'hidden' }}
            />
            {!isCameraRunning && (
              <div className="preview-placeholder">
                <CiCamera />
                <p>Start the camera to view the live feed.</p>
              </div>
            )}
          </div>
        </section>
      </div>

      {uiMessage && (
        <div className="toast">
          <span>{uiMessage}</span>
          <button onClick={() => setUiMessage(null)}>Dismiss</button>
        </div>
      )}
    </div>
  )
}

export default App