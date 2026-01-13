import './App.css'
import Orb from './components/orb'
import { useState, useEffect } from 'react'
import { CiCamera, CiMicrophoneOn } from 'react-icons/ci'
import { FiImage, FiMic, FiMicOff, FiStopCircle, FiActivity } from 'react-icons/fi'
import { cameraService } from './services/cameraService'
import { voiceService } from './services/voiceService'

function App() {
  const [isCameraRunning, setIsCameraRunning] = useState(false)
  const [cameraFrameUrl, setCameraFrameUrl] = useState(null)
  const [sceneAnalysis, setSceneAnalysis] = useState('')
  const [analysisLoading, setAnalysisLoading] = useState(false)
  const [voiceResult, setVoiceResult] = useState(null)
  const [isRecording, setIsRecording] = useState(false)
  const [recordingMode, setRecordingMode] = useState(null)
  const [isSpeechActive, setIsSpeechActive] = useState(false)
  const [uiMessage, setUiMessage] = useState(null)
  const [latencyMetrics, setLatencyMetrics] = useState(null)

  useEffect(() => {
    checkCameraStatus()
  }, [])

  useEffect(() => {
    if (!isCameraRunning) {
      setCameraFrameUrl(null)
      return
    }
    const interval = setInterval(() => {
      setCameraFrameUrl(cameraService.getFrameUrl())
    }, 150)
    return () => clearInterval(interval)
  }, [isCameraRunning])

  useEffect(() => {
    if (!isRecording && !isSpeechActive) return
    const interval = setInterval(async () => {
      try {
        const status = await voiceService.getVoiceStatus()
        if (typeof status.is_speaking === 'boolean') {
          setIsSpeechActive(status.is_speaking)
        }
        if (typeof status.is_recording === 'boolean' && !status.is_recording) {
          setIsRecording(false)
          setRecordingMode(null)
        }
      } catch (error) {
        console.error('Error fetching voice status:', error)
      }
    }, 1200)
    return () => clearInterval(interval)
  }, [isRecording, isSpeechActive])

  const checkCameraStatus = async () => {
    try {
      const status = await cameraService.getCameraStatus()
      setIsCameraRunning(status.is_running)
    } catch (error) {
      console.error('Error checking camera status:', error)
      setUiMessage('Unable to reach camera service. Is the Flask server running?')
    }
  }

  const handleCameraToggle = async () => {
    try {
      if (isCameraRunning) {
        await cameraService.stopCamera()
        setIsCameraRunning(false)
      } else {
        const result = await cameraService.startCamera()
        if (result.status === 'success') {
          setIsCameraRunning(true)
        } else {
          setUiMessage(result.message || 'Unable to start camera')
        }
      }
    } catch (error) {
      console.error('Error toggling camera:', error)
      setUiMessage('Error talking to camera service. Check that the backend is running.')
    }
  }

  const handleAnalyze = async (shouldSpeak = false) => {
    setAnalysisLoading(true)
    try {
      const result = await cameraService.analyzeScene({ speak: shouldSpeak })
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
      setUiMessage('Microphone error. Check permissions and backend logs.')
    }
  }

  const handleVoiceStop = async () => {
    try {
      setUiMessage('Processing...')
      const result = await voiceService.stopVoice()
      console.log('Voice stop result:', result)
      console.log('Metrics received:', result.metrics)
      setIsRecording(false)
      setRecordingMode(null)
      if (result.status === 'success' && result.data) {
        setVoiceResult(result.data)
        setIsSpeechActive(true)
        // Store the metrics - they're in the result directly
        if (result.metrics) {
          console.log('Setting latency metrics:', result.metrics)
          setLatencyMetrics(result.metrics)
        }
        setUiMessage(null)
      } else {
        setUiMessage(result.message || 'No response generated.')
      }
    } catch (error) {
      console.error('Error stopping voice recording:', error)
      setUiMessage('Could not process the recording. Please try again.')
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
              <p className="eyebrow">MDN Assist</p>
              <h1>Hello Steve</h1>
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

          {/* Latency Metrics Card */}
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
            {isCameraRunning && cameraFrameUrl ? (
              <img src={cameraFrameUrl} alt="Camera feed" />
            ) : (
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
