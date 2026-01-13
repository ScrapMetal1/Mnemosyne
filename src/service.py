# Disable MKL to avoid OpenMP runtime conflicts on Windows
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import queue
import threading
import io
import base64
import re
import logging

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from openai import OpenAI


load_dotenv()

client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
CORS(app)

# Suppress Flask/Werkzeug request logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

# ==== AUDIO CONFIG ===========================================================
SAMPLE_RATE = 16000
CHANNELS = 1
FILENAME = "temp_recording.wav"

# ==== GLOBAL STATE ===========================================================
audio_queue = queue.Queue()
current_stream = None
current_mode = "plain"  # 'plain' or 'scene'
response_lock = threading.Lock()
current_response = None

is_speaking = False
playback_stream = None


def _elapsed_ms(start_time):
    return (time.perf_counter() - start_time) * 1000


def record_metric(metrics, key, start_time=None, value=None):
    if metrics is None:
        return
    if start_time is not None:
        metrics[key] = _elapsed_ms(start_time)
    elif value is not None:
        metrics[key] = value


def log_latency(stage, start_time, suffix=""):
    """Print lightweight latency metrics in milliseconds."""
    elapsed_ms = _elapsed_ms(start_time)
    message = f"[LATENCY] {stage}: {elapsed_ms:.1f} ms"
    if suffix:
        message = f"{message} {suffix}"
    print(message)


def call_openai_vision_api(frame, user_question=None, use_voice=False, metrics=None):
    """
    Send current camera frame to OpenAI Vision API for scene analysis.
    """
    try:
        encode_start = time.perf_counter()
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        image_data_url = f"data:image/jpeg;base64,{frame_base64}"
        encode_ms = _elapsed_ms(encode_start)
        
        if user_question:
            record_metric(metrics, "vision_encode_ms", value=encode_ms)
            print("Thinking with vision context...")
        else:
            record_metric(metrics, "image_encode_ms", value=encode_ms)
        
        if user_question:
            text_prompt = f"The user asked: \"{user_question}\"\n\nAnswer their question based on what you see in this image. Be concise and helpful."
            system_prompt = "You are a helpful assistant for a vision-impaired user. Answer their question using what you see in the image. Be concise and helpful."
            log_label = "OpenAI vision contextual response"
            metric_key = "vision_contextual_response_ms"
            total_metric_key = "vision_contextual_total_ms"
        else:
            text_prompt = "Describe this scene concisely for a vision-impaired user. Focus on objects, people, text, and spatial layout."
            system_prompt = None
            log_label = "OpenAI Vision API"
            metric_key = "vision_api_ms"
            total_metric_key = "vision_api_total_ms"
        
        api_start = time.perf_counter()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}}
            ]
        })
        
        response = client_ai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            max_tokens=200,
        )
        
        analysis = response.choices[0].message.content.strip()
        elapsed_ms = _elapsed_ms(api_start)
        
        log_latency(log_label, api_start, f"(image_size={len(frame_base64)} bytes)")
        record_metric(metrics, metric_key, value=elapsed_ms)
        record_metric(metrics, total_metric_key, value=encode_ms + elapsed_ms)
        
        if use_voice:
            speak(analysis, metrics=metrics)
        
        return analysis
        
    except Exception as exc:
        error_msg = f"Error calling Vision API: {exc}"
        log_latency("OpenAI Vision API", api_start if 'api_start' in locals() else time.perf_counter(), "ERROR")
        if metrics:
            if user_question:
                record_metric(metrics, "vision_contextual_error", value=str(exc))
            else:
                record_metric(metrics, "vision_api_error", value=str(exc))
        if use_voice:
            speak(error_msg, metrics=metrics)
        return error_msg


# ==== VOICE / AUDIO HELPERS ==================================================
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())


def start_recording():
    audio_queue.queue.clear()
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        callback=audio_callback,
    )
    stream.start()
    print("Recording started...")
    return stream


def stop_recording(stream):
    print("Recording stopped. Saving file...")
    stream.stop()
    stream.close()

    frames = []
    while not audio_queue.empty():
        frames.append(audio_queue.get())

    if not frames:
        print("No audio data captured.")
        return None

    audio = np.concatenate(frames, axis=0)
    sf.write(FILENAME, audio, SAMPLE_RATE, subtype="PCM_16")
    print("Saved to", FILENAME)
    return FILENAME


def transcribe_audio(filename, metrics=None):
    print("Transcribing...")
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 1000
    recognizer.dynamic_energy_threshold = True

    with sr.AudioFile(filename) as source:
        audio_data = recognizer.record(source)
    try:
        transcribe_start = time.perf_counter()
        text = recognizer.recognize_google(audio_data)
        elapsed_ms = _elapsed_ms(transcribe_start)
        log_latency("Transcription", transcribe_start)
        record_metric(metrics, "transcription_ms", value=elapsed_ms)
        print(f"Transcription complete: {text}")
        return text
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return ""
    except sr.RequestError as exc:
        print(f"Google Speech API error: {exc}")
        return ""


def _synthesize_speech(text):
    tts_start = time.perf_counter()
    audio_response = client_ai.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
    )
    buffer = io.BytesIO(audio_response.content)
    data, samplerate = sf.read(buffer)
    generation_ms = _elapsed_ms(tts_start)
    playback_estimate_ms = len(data) / samplerate * 1000
    log_latency("TTS synthesis", tts_start, f"(chars={len(text)})")
    return data, samplerate, generation_ms, playback_estimate_ms


def _playback_worker(audio_data, samplerate):
    global is_speaking, playback_stream
    try:
        is_speaking = True
        playback_stream = sd.play(audio_data, samplerate)
        if playback_stream is not None:
            while is_speaking and playback_stream.active:
                time.sleep(0.05)
        else:
            est_duration = len(audio_data) / samplerate
            start_time = time.time()
            while is_speaking and (time.time() - start_time < est_duration + 1):
                time.sleep(0.05)
    except Exception as exc:
        print(f"Voice playback error: {exc}")
    finally:
        is_speaking = False
        playback_stream = None


def speak(text, metrics=None):
    print("Speaking...")
    try:
        data, samplerate, gen_ms, playback_estimate_ms = _synthesize_speech(text)
        if metrics is not None:
            metrics["tts_generation_ms"] = gen_ms
            metrics["tts_playback_estimate_ms"] = playback_estimate_ms
        playback_thread = threading.Thread(
            target=_playback_worker, args=(data, samplerate), daemon=True
        )
        playback_thread.start()
    except Exception as exc:
        if metrics is not None:
            metrics["tts_generation_error"] = str(exc)
        print(f"Voice synthesis error: {exc}")


def stop_speech():
    global is_speaking, playback_stream
    if not is_speaking:
        return
    print("Stopping speech playback...")
    is_speaking = False
    sd.stop()
    if playback_stream is not None:
        try:
            playback_stream.stop()
            playback_stream.close()
        except Exception:
            pass
        playback_stream = None
    print("Speech interrupted")


def get_ai_response(prompt, metrics=None):
    print("Thinking...")
    try:
        ai_start = time.perf_counter()
        response = client_ai.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": "You are a helpful personal assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        elapsed_ms = _elapsed_ms(ai_start)
        log_latency("OpenAI voice response", ai_start)
        record_metric(metrics, "openai_response_ms", value=elapsed_ms)
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"Error getting AI response: {exc}"


def get_ai_response_streaming(prompt):
    """Returns a streaming generator for LLM response."""
    print("Thinking (streaming)...")
    try:
        stream = client_ai.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": "You are a helpful personal assistant."},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        return stream
    except Exception as e:
        print(f"[ERROR] Error creating stream: {e}")
        raise


def get_ai_response_with_vision(voice_text, frame, metrics=None):
    """Get AI response using Vision API with user's voice question and current frame."""
    return call_openai_vision_api(frame, user_question=voice_text, use_voice=False, metrics=metrics)


def get_ai_response_with_vision_streaming(voice_text, frame):
    """Returns a streaming generator for Vision API response."""
    print("Thinking with vision context (streaming)...")
    
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    image_data_url = f"data:image/jpeg;base64,{frame_base64}"
    
    text_prompt = f"The user asked: \"{voice_text}\"\n\nAnswer their question based on what you see in this image. Be concise and helpful."
    
    try:
        stream = client_ai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant for a vision-impaired user. Answer their question using what you see in the image. Be concise and helpful."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}}
                    ]
                }
            ],
            max_tokens=200,
            stream=True,
        )
        return stream
    except Exception as e:
        print(f"[ERROR] Error creating vision stream: {e}")
        raise


# ==== STREAMING AUDIO PIPELINE WITH LATENCY TRACKING =========================
class StreamingAudioPipeline: 
    def __init__(self, client_ai, pipeline_start_time):
        self.client = client_ai
        self.sentence_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.full_response_text = []
        self.is_active = False
        self.threads = []
        
        # Latency tracking
        self.pipeline_start = pipeline_start_time
        self.metrics = {}
        self.first_token_time = None
        self.first_sentence_time = None
        self.first_tts_complete_time = None
        self.first_audio_play_time = None
        self.text_complete_time = None

    def start(self, llm_stream_generator):
        print("[STREAM] Starting streaming audio pipeline...")
        self.is_active = True
        self.stop_event.clear()

        t_text = threading.Thread(target=self._process_text, args=(llm_stream_generator,), daemon=True)
        t_tts = threading.Thread(target=self._generate_audio, daemon=True)
        t_play = threading.Thread(target=self._play_audio, daemon=True)

        self.threads = [t_text, t_tts, t_play]
        for thread in self.threads:
            thread.start()
        print("[STREAM] All threads started")

    def stop(self):
        self.stop_event.set()
        self.is_active = False
        with self.sentence_queue.mutex:
            self.sentence_queue.queue.clear()
        with self.audio_queue.mutex: 
            self.audio_queue.queue.clear()
        sd.stop()

    def wait_for_completion(self):
        """Wait for all threads and return full response text and metrics."""
        for thread in self.threads:
            thread.join()
        return "".join(self.full_response_text), self.get_metrics()

    def get_metrics(self):
        """Calculate and return all latency metrics."""
        metrics = {}
        
        # Time to first LLM token (from pipeline start)
        if self.first_token_time:
            metrics["llm_first_token_ms"] = _elapsed_ms(self.pipeline_start) - _elapsed_ms(self.first_token_time) + _elapsed_ms(self.first_token_time)
            metrics["llm_first_token_ms"] = (self.first_token_time - self.pipeline_start) * 1000
        
        # Time to first complete sentence
        if self.first_sentence_time:
            metrics["first_sentence_ms"] = (self.first_sentence_time - self.pipeline_start) * 1000
        
        # Time to first TTS audio generated
        if self.first_tts_complete_time:
            metrics["first_tts_ready_ms"] = (self.first_tts_complete_time - self.pipeline_start) * 1000
        
        # ⭐ THE KEY METRIC: Time from pipeline start to first audio playback
        if self.first_audio_play_time:
            metrics["time_to_first_audio_ms"] = (self.first_audio_play_time - self.pipeline_start) * 1000
        
        # Total text generation time
        if self.text_complete_time:
            metrics["total_text_generation_ms"] = (self.text_complete_time - self.pipeline_start) * 1000
        
        return metrics

    def _process_text(self, generator):
        """Consumes LLM tokens, buffers them, and puts complete sentences into the queue."""
        buffer = ""
        token_count = 0
        print("[STREAM] Text processing started...")
        try:
            for chunk in generator:
                if self.stop_event.is_set(): 
                    break
                if not chunk.choices:
                    continue
                
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue
                
                token_count += 1
                if token_count == 1:
                    self.first_token_time = time.perf_counter()
                    ttft = (self.first_token_time - self.pipeline_start) * 1000
                    print(f"[STREAM] ⚡ First token received at {ttft:.0f}ms")
                
                self.full_response_text.append(delta)
                buffer += delta

                parts = re.split(r'(?<=[.?!])\s+', buffer)
                
                if len(parts) > 1:
                    for part in parts[:-1]:
                        if part.strip():
                            if self.first_sentence_time is None:
                                self.first_sentence_time = time.perf_counter()
                                fst = (self.first_sentence_time - self.pipeline_start) * 1000
                                print(f"[STREAM] ⚡ First sentence ready at {fst:.0f}ms")
                            self.sentence_queue.put(part.strip())
                    buffer = parts[-1]
            
            if buffer.strip() and not self.stop_event.is_set():
                if self.first_sentence_time is None:
                    self.first_sentence_time = time.perf_counter()
                    fst = (self.first_sentence_time - self.pipeline_start) * 1000
                    print(f"[STREAM] ⚡ First sentence ready (flush) at {fst:.0f}ms")
                self.sentence_queue.put(buffer.strip())
            
            self.text_complete_time = time.perf_counter()
            total_text = (self.text_complete_time - self.pipeline_start) * 1000
            print(f"[STREAM] Text complete at {total_text:.0f}ms (tokens: {token_count})")
        except Exception as e:
            print(f"[STREAM] Text processing error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.sentence_queue.put(None)

    def _generate_audio(self):
        """Consumes sentences and calls TTS API."""
        print("[STREAM] TTS generation started...")
        sentence_count = 0
        while not self.stop_event.is_set():
            text = self.sentence_queue.get()
            if text is None: 
                self.audio_queue.put(None)
                break
            
            sentence_count += 1
            try:
                tts_start = time.perf_counter()
                response = self.client.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input=text,
                    response_format="pcm"
                )
                
                if self.first_tts_complete_time is None:
                    self.first_tts_complete_time = time.perf_counter()
                    ftt = (self.first_tts_complete_time - self.pipeline_start) * 1000
                    tts_dur = (self.first_tts_complete_time - tts_start) * 1000
                    print(f"[STREAM] ⚡ First TTS ready at {ftt:.0f}ms (TTS took {tts_dur:.0f}ms)")
                
                self.audio_queue.put(response.content)
            except Exception as e:
                print(f"[STREAM] TTS error: {e}")
        print(f"[STREAM] TTS complete (sentences: {sentence_count})")

    def _play_audio(self):
        """Consumes audio chunks and plays them."""
        print("[STREAM] Audio playback started...")
        chunk_count = 0
        try:
            with sd.OutputStream(samplerate=24000, channels=1, dtype='int16') as stream:
                while not self.stop_event.is_set():
                    chunk = self.audio_queue.get()
                    if chunk is None: 
                        break
                    
                    chunk_count += 1
                    if chunk_count == 1:
                        self.first_audio_play_time = time.perf_counter()
                        ttfa = (self.first_audio_play_time - self.pipeline_start) * 1000
                        print(f"[STREAM] 🔊 FIRST AUDIO PLAYBACK at {ttfa:.0f}ms")
                    
                    data = np.frombuffer(chunk, dtype=np.int16)
                    stream.write(data)
        except Exception as e:
            print(f"[STREAM] Playback error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_active = False
            print(f"[STREAM] Playback complete (chunks: {chunk_count})")


# Global pipeline instance
pipeline = None


# ==== CAMERA SERVICE =========================================================
class CameraService:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.processing_thread = None
        self.lock = threading.Lock()
        self.current_frame = None

    def start_camera(self, cam_index=0):
        if self.is_running:
            return {"status": "success", "message": "Camera already running"}

        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            return {"status": "error", "message": f"Could not open camera index {cam_index}"}

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cv2.setUseOptimized(True)

        self.cap = cap
        self.is_running = True

        self.processing_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.processing_thread.start()
        return {"status": "success", "message": "Camera started"}

    def stop_camera(self):
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            self.processing_thread = None
        if self.cap:
            self.cap.release()
            self.cap = None
        with self.lock:
            self.current_frame = None
        return {"status": "success", "message": "Camera stopped"}

    def _process_frames(self):
        fps = 0.0
        start_time = time.time()

        while self.is_running and self.cap:
            ok, frame = self.cap.read()
            if not ok:
                print("Frame grab failed.")
                continue

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - start_time, 1e-6))
            start_time = now
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1, cv2.LINE_AA)

            with self.lock:
                self.current_frame = frame.copy()

    def get_current_frame(self):
        with self.lock:
            if self.current_frame is None:
                return None
            _, buffer = cv2.imencode(".jpg", self.current_frame)
            return buffer.tobytes()

    def analyze_scene(self, speak_result=False, metrics=None):
        """Analyze current frame with Vision API."""
        with self.lock:
            if self.current_frame is None:
                message = "No frame available for analysis."
                if speak_result:
                    speak(message, metrics=metrics)
                return message
            frame = self.current_frame.copy()
        return call_openai_vision_api(frame, use_voice=speak_result, metrics=metrics)


camera_service = CameraService()


# ==== ROUTES: CAMERA =========================================================
@app.route("/api/camera/start", methods=["POST"])
def start_camera():
    op_start = time.perf_counter()
    result = camera_service.start_camera()
    metrics = {"operation_ms": _elapsed_ms(op_start)}
    if isinstance(result, dict):
        result = {**result, "metrics": metrics}
    else:
        result = {"status": "error", "message": "Unexpected response", "metrics": metrics}
    status_code = 200 if result.get("status") == "success" else 400
    return jsonify(result), status_code


@app.route("/api/camera/stop", methods=["POST"])
def stop_camera():
    op_start = time.perf_counter()
    result = camera_service.stop_camera()
    metrics = {"operation_ms": _elapsed_ms(op_start)}
    result = {**result, "metrics": metrics}
    return jsonify(result)


@app.route("/api/camera/status", methods=["GET"])
def camera_status():
    return jsonify({"is_running": camera_service.is_running, "metrics": {"operation_ms": 0.0}})


@app.route("/api/camera/frame", methods=["GET"])
def get_frame():
    frame = camera_service.get_current_frame()
    if frame is None:
        return jsonify({"status": "error", "message": "No frame available"}), 404
    return Response(frame, mimetype="image/jpeg")


@app.route("/api/scene/analyze", methods=["POST"])
def analyze_scene():
    payload = request.get_json(silent=True) or {}
    speak_flag = bool(payload.get("speak"))
    metrics = {}
    analysis = camera_service.analyze_scene(speak_result=speak_flag, metrics=metrics)
    return jsonify({"analysis": analysis, "spoken": speak_flag, "metrics": metrics})


# ==== ROUTES: VOICE ==========================================================
@app.route("/api/voice/start", methods=["POST"])
def start_voice():
    global current_stream, current_mode
    metrics = {}
    if current_stream is not None:
        return jsonify({
            "status": "error",
            "message": "Voice recording already in progress.",
            "metrics": metrics,
        }), 400

    payload = request.get_json(silent=True) or {}
    mode = payload.get("mode", "plain")
    if mode not in ("plain", "scene"):
        return jsonify({
            "status": "error",
            "message": "Invalid mode supplied.",
            "metrics": metrics
        }), 400

    if is_speaking:
        stop_speech()

    try:
        op_start = time.perf_counter()
        current_stream = start_recording()
        current_mode = mode
        record_metric(metrics, "operation_ms", start_time=op_start)
        return jsonify({
            "status": "success",
            "message": "Voice recording started.",
            "mode": mode,
            "metrics": metrics,
        })
    except Exception as exc:
        current_stream = None
        metrics["operation_error"] = str(exc)
        return jsonify({
            "status": "error",
            "message": f"Could not access microphone: {exc}",
            "metrics": metrics,
        }), 500


@app.route("/api/voice/stop", methods=["POST"])
def stop_voice():
    global current_stream, current_response, pipeline
    
    # This is the critical moment - pipeline starts when recording stops
    pipeline_start = time.perf_counter()
    
    if current_stream is None:
        return jsonify({
            "status": "error",
            "message": "No voice recording in progress.",
            "metrics": {"pipeline_total_ms": 0.0},
        }), 400

    metrics = {}
    stream = current_stream
    current_stream = None
    
    # Stage 1: Save audio file
    save_start = time.perf_counter()
    filename = stop_recording(stream)
    metrics["audio_save_ms"] = _elapsed_ms(save_start)

    if not filename:
        metrics["pipeline_total_ms"] = _elapsed_ms(pipeline_start)
        return jsonify({
            "status": "error",
            "message": "No audio data captured.",
            "metrics": metrics
        }), 400

    # Stage 2: Transcription
    transcribe_start = time.perf_counter()
    user_text = transcribe_audio(filename, metrics=metrics)
    if os.path.exists(FILENAME):
        os.remove(FILENAME)

    if not user_text:
        metrics["pipeline_total_ms"] = _elapsed_ms(pipeline_start)
        return jsonify({
            "status": "error",
            "message": "Could not understand audio.",
            "metrics": metrics,
        }), 200

    # Stage 3: Prepare LLM request
    frame = None
    llm_generator = None
    
    if current_mode == "scene":
        with camera_service.lock:
            frame = camera_service.current_frame.copy() if camera_service.current_frame is not None else None
        
        if frame is not None:
            llm_generator = get_ai_response_with_vision_streaming(user_text, frame)
        else:
            llm_generator = get_ai_response_streaming(user_text)
    else:
        llm_generator = get_ai_response_streaming(user_text)

    # Stage 4: Run streaming pipeline with latency tracking
    pipeline = StreamingAudioPipeline(client_ai, pipeline_start)
    pipeline.start(llm_generator)
    
    # Wait for completion and get metrics
    response_text, stream_metrics = pipeline.wait_for_completion()
    
    # Merge stream metrics into main metrics
    metrics.update(stream_metrics)
    
    # Calculate total pipeline time
    metrics["pipeline_total_ms"] = _elapsed_ms(pipeline_start)

    payload = {
        "transcript": user_text,
        "response": response_text,
        "mode": current_mode,
        "context": "vision" if (current_mode == "scene" and frame is not None) else "plain",
    }

    with response_lock:
        current_response = payload

    # Log summary
    print(f"\n{'='*60}")
    print(f"📊 PIPELINE METRICS SUMMARY")
    print(f"{'='*60}")
    if "time_to_first_audio_ms" in metrics:
        print(f"⭐ TIME TO FIRST AUDIO: {metrics['time_to_first_audio_ms']:.0f}ms")
    print(f"   Audio save: {metrics.get('audio_save_ms', 0):.0f}ms")
    print(f"   Transcription: {metrics.get('transcription_ms', 0):.0f}ms")
    if "llm_first_token_ms" in metrics:
        print(f"   LLM first token: {metrics['llm_first_token_ms']:.0f}ms")
    if "first_sentence_ms" in metrics:
        print(f"   First sentence: {metrics['first_sentence_ms']:.0f}ms")
    if "first_tts_ready_ms" in metrics:
        print(f"   First TTS ready: {metrics['first_tts_ready_ms']:.0f}ms")
    print(f"   Total pipeline: {metrics['pipeline_total_ms']:.0f}ms")
    print(f"{'='*60}\n")

    return jsonify({
        "status": "success",
        "message": "Voice recording processed.",
        "data": payload,
        "metrics": metrics,
    })


@app.route("/api/voice/response", methods=["GET"])
def get_voice_response():
    with response_lock:
        if current_response is None:
            return jsonify({"status": "error", "message": "No response available."}), 404
        return jsonify({"status": "success", "data": current_response})


@app.route("/api/voice/status", methods=["GET"])
def voice_status():
    speaking = is_speaking or (pipeline is not None and pipeline.is_active)
    return jsonify({
        "is_recording": current_stream is not None,
        "is_speaking": speaking,
        "metrics": {"operation_ms": 0.0},
    })


@app.route("/api/voice/stop-speech", methods=["POST"])
def api_stop_speech():
    global pipeline
    if pipeline is not None and pipeline.is_active:
        pipeline.stop()
    stop_speech()
    return jsonify({
        "status": "success",
        "message": "Speech playback stopped.",
        "metrics": {"operation_ms": 0.0},
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
