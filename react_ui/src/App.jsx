import './App.css'
import { useState, useEffect, useRef } from 'react'
import { cameraService } from './services/cameraService'
import { voiceService } from './services/voiceService'
import { memoryService } from './services/memoryService'

import statusBg from './assets/status.svg'
import listeningBg from './assets/listeningcont.svg'
import speakBg from './assets/speakcont.svg'
import sceneControlBg from './assets/scenecontrolcont.svg'
import sceneItemBg from './assets/scenecontitem.svg'
import cameraIcon from './assets/camera.svg'
import micIcon from './assets/microphone.svg'
import describeIcon from './assets/describe.svg'
import descNSpeakIcon from './assets/descnspeak.svg'
import memoryIcon from './assets/memory.svg'

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
        setLatencyMetrics(null)
        setUiMessage(null)
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
          playNextAudio();
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

  const handleSaveMemory = async () => {
    setMemoryLoading(true)
    try {
      const imageBase64 = captureFrame();
      if (!imageBase64) {
        throw new Error("No camera frame available");
      }
      const result = await memoryService.captureMemory(imageBase64)
      if (result.status === 'success') {
        setUiMessage(`Memory saved: ${result.description}`)
      } else {
        setUiMessage(`Error: ${result.message}`)
      }
    } catch (err) {
      console.error("Save memory error:", err);
      setUiMessage(`Could not save memory: ${err.message || "Unknown error"}`)
    } finally {
      setMemoryLoading(false)
    }
  }

  const aiDisplayText = (voiceResult?.response && voiceResult.response.trim())
    || sceneAnalysis
    || ''

  const handleSpeakToggle = () => {
    if (isRecording && recordingMode === 'plain') {
      handleVoiceStop()
    } else {
      handleVoiceStart('plain')
    }
  }

  return (
    <div className="app-root">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className={`camera-bg${isCameraRunning ? ' visible' : ''}`}
        onLoadedMetadata={() => videoRef.current && videoRef.current.play()}
      />

      <div className="overlay">
        {/* Top Left: Welcome + Status + AI response */}
        <div className="top-left">
          <h1 className="welcome-title">Welcome Back, Rez</h1>
          <p className="status-label">Status:</p>
          <div
            className="status-cont"
            style={{ backgroundImage: `url(${statusBg})` }}
          >
            <button
              className={`icon-btn${isCameraRunning ? ' active' : ''}`}
              onClick={handleCameraToggle}
              title="Toggle Camera"
            >
              <img src={cameraIcon} alt="Camera" />
            </button>
            <button
              className={`icon-btn${isRecording ? ' active' : ''}`}
              onClick={handleSpeakToggle}
              title="Toggle Microphone"
            >
              <img src={micIcon} alt="Microphone" />
            </button>
          </div>
          {aiDisplayText && (
            <p className="ai-response">{aiDisplayText}</p>
          )}
        </div>

        {/* Top Right: Listening container — fades in when recording */}
        <div
          className={`listening-cont${isRecording ? ' visible' : ''}`}
          style={{ backgroundImage: `url(${listeningBg})` }}
        >
          <p className="listening-title">Listening...</p>
          <p className="transcript-text">
            {voiceResult?.transcript && voiceResult.transcript !== '...'
              ? voiceResult.transcript
              : ''}
          </p>
        </div>

        {/* Bottom Left: Speak button */}
        <button
          className={`speak-btn${isRecording && recordingMode === 'plain' ? ' active' : ''}`}
          style={{ backgroundImage: `url(${speakBg})` }}
          onClick={handleSpeakToggle}
        >
          <img src={micIcon} alt="Microphone" />
          <span>Speak</span>
        </button>

        {/* Bottom Right: Scene Controls */}
        <div
          className="scene-controls-cont"
          style={{ backgroundImage: `url(${sceneControlBg})` }}
        >
          <p className="scene-title">Scene Controls</p>

          <button
            className="scene-item-btn"
            style={{ backgroundImage: `url(${sceneItemBg})` }}
            onClick={() => handleAnalyze(false)}
            disabled={!isCameraRunning || analysisLoading}
          >
            <img src={describeIcon} alt="" />
            <span>Describe Scene</span>
          </button>

          <button
            className="scene-item-btn"
            style={{ backgroundImage: `url(${sceneItemBg})` }}
            onClick={() => handleAnalyze(true)}
            disabled={!isCameraRunning || analysisLoading}
          >
            <img src={descNSpeakIcon} alt="" />
            <span>Describe &amp; Speak</span>
          </button>

          <button
            className="scene-item-btn"
            style={{ backgroundImage: `url(${sceneItemBg})` }}
            onClick={handleSaveMemory}
            disabled={!isCameraRunning || memoryLoading}
          >
            <img src={memoryIcon} alt="" />
            <span>Save Memory</span>
          </button>
        </div>
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
