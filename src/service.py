# Disable MKL to avoid OpenMP runtime conflicts on Windows
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import base64
import logging
import cv2
import numpy as np
import json
import threading
import queue
import concurrent.futures
import re

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS
from openai import OpenAI

from embedding_storage import EmbeddingStore, EmbeddingExtractor
from fastvlm_inference import describe_frame, _load_fastvlm
from filtering import requires_memory_recall, extract_time_filter
 

load_dotenv()

client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
CORS(app)

# Suppress Flask/Werkzeug request logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

#instantiate the classes
extractor = EmbeddingExtractor()
storage = EmbeddingStore()

# Ensure the image storage directory exists
IMAGE_DIR = os.path.join(storage.storage_dir, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

#load fastvlm
_load_fastvlm()

def _elapsed_ms(start_time):
    return (time.perf_counter() - start_time) * 1000

def record_metric(metrics, key, start_time=None, value=None):
    if metrics is None:
        return
    if start_time is not None:
        metrics[key] = _elapsed_ms(start_time)
    elif value is not None:
        metrics[key] = value

def base64_to_cv2(base64_string):
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def call_openai_vision_api(frame, user_question=None, metrics=None):
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
        else:
            text_prompt = "Describe this scene concisely for a vision-impaired user. Focus on objects, people, text, and spatial layout."
            system_prompt = "You are a helpful assistant."
            
        api_start = time.perf_counter()
        messages = [{"role": "system", "content": system_prompt}, {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}}
            ]
        }]
        
        # Note: the original used gpt-4.1-mini. Using gpt-4o-mini as fallback if needed.
        try:
            response = client_ai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=200,
            )
        except Exception:
            response = client_ai.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=messages,
                max_tokens=200,
            )
        
        analysis = response.choices[0].message.content.strip()
        elapsed_ms = _elapsed_ms(api_start)
        record_metric(metrics, "vision_api_ms", value=elapsed_ms)
        return analysis
        
    except Exception as exc:
        error_msg = f"Error calling Vision API: {exc}"
        print(error_msg)
        return "I couldn't process the image."


def get_ai_response_streaming(prompt):
    """Returns a streaming generator for LLM response."""
    print("Thinking (streaming)...")
    
    # Number of memories to retrieve. Defaults to 5 in the underlying EmbeddingStore if not provided.
    top_k = 1

    if requires_memory_recall(prompt):
        print(f"\n[DEBUG] Memory recall triggered for prompt: '{prompt}'")
        
        #Find a start time to Filter our for time
        start_time = extract_time_filter(prompt)
        if start_time:
            print(f"[DEBUG] Applied time filter: > {start_time}")
        
        #Embed the prompt
        prompt_embedding = extractor.extract_embeddings(prompt)

        #Search for relvent memories filtering out distant memories if requried. 
        if start_time:
            relevant_memories = storage.search(prompt_embedding, top_k=top_k, filter={'start_time': start_time})
        else:
            relevant_memories = storage.search(prompt_embedding, top_k=top_k)

        print(f"[DEBUG] Found {len(relevant_memories)} relevant memories.")

        context_str = ""
        retrieved_images = [] # We'll store any base64 images we find here
        
        for i, mem in enumerate(relevant_memories):
            print(f"  [{i+1}] Memory ID: {mem['id']} | Similarity: {mem.get('similarity', 0.0):.4f} | Text: '{mem['text']}'")
            context_str += f"- {mem['text']} (at {mem['timestamp']})\n"
            
            # 1. Look for the corresponding image on disk using the memory's ID
            image_path = os.path.join(IMAGE_DIR, f"{mem['id']}.jpg")
            if os.path.exists(image_path):
                print(f"      -> Found corresponding image on disk")
                # 2. If the file exists, read it from the disk
                img = cv2.imread(image_path)
                
                # 3. Convert the cv2 image back into a base64 string so we can send it to OpenAI
                _, buffer = cv2.imencode('.jpg', img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                retrieved_images.append(img_base64)
            else:
                print(f"      -> No image found on disk for this memory")

        system_prompt = "You are a helpful personal assistant with access to past memories. Be concise in your response"
        if context_str:
            system_prompt += f"\n\nCONTEXT FROM MEMORY:\n{context_str}"
    else: 
        print(f"\n[DEBUG] Memory recall NOT triggered for prompt: '{prompt}'")
        system_prompt = "You are a helpful personal assistant. Be concise in your response"
        retrieved_images = []

    # Prepare the user content payload
    # By default, we always send the text prompt
    user_content = [{"type": "text", "text": prompt}]
    
    # If we retrieved any images from memory, attach them to the payload!
    for img_b64 in retrieved_images:
        user_content.append({
            "type": "image_url", 
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })

    try:
        stream = client_ai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                # We send the array of text+images if we have images, otherwise just send the prompt string
                {"role": "user", "content": user_content if len(user_content) > 1 else prompt},
            ],
            stream=True,
        )
        return stream
    except Exception as e:
        print(f"[ERROR] Error creating stream: {e}")
        raise

def get_ai_response_with_vision_streaming(voice_text, frame):
    """Returns a streaming generator for Vision API response."""
    print("Thinking with vision context (streaming)...")
    
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    image_data_url = f"data:image/jpeg;base64,{frame_base64}"
    
    text_prompt = f"The user asked: \"{voice_text}\"\n\nAnswer their question based on what you see in this image. Be concise and helpful."
    
    try:
        stream = client_ai.chat.completions.create(
            model="gpt-4o-mini",
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

## MEMORY FUNCTIONS

def capture_memory_snapshot(frame_base64):
    try:        
        frame = base64_to_cv2(frame_base64)
        
        ## DEBUG: Save the frame to disk to verify what the backend is seeing
        #cv2.imwrite("debug_latest_memory.jpg", frame)

        describe_start = time.time()
        description = describe_frame(frame)  #fastvlm inference
        describe_end = time.time()
        print(f"FastVLM describe time: {(describe_end-describe_start) * 1000:.2f}ms")
        
        embed_start = time.time()
        embedding = extractor.extract_embeddings(description)
        embed_end = time.time()
        
        store_start = time.time()
        #Save the embedding and get the unique memory_id (e.g. memory_17000000_1)
        memory_id = storage.add(embedding=embedding, text=description) 
        
        #Save the image to the disk using the memory_id as the filename.
        #This writes the cv2 'frame' out as a .jpg file in the IMAGE_DIR folder we made at the top.
        image_path = os.path.join(IMAGE_DIR, f"{memory_id}.jpg")
        cv2.imwrite(image_path, frame)
        store_end = time.time()

        return {
            "status": "success",
            "description": description, 
            "FastVLM describe time": (describe_end - describe_start) * 1000,
            "Embedding_time_ms": (embed_end - embed_start) * 1000,
            "Store Time:": (store_end - store_start) * 1000
        }

    except Exception as e:
        print(f"Manual Memory Capture Error: {e}")
        return {"status": "error", "message": str(e)}

# ==== ROUTES: CAMERA =========================================================

# The frontend manages the camera locally now. These stub endpoints are left just in case.
@app.route("/api/camera/start", methods=["POST"])
def start_camera():
    return jsonify({"status": "success", "message": "Handled locally by frontend", "metrics": {}})

@app.route("/api/camera/stop", methods=["POST"])
def stop_camera():
    return jsonify({"status": "success", "message": "Handled locally by frontend", "metrics": {}})

@app.route("/api/camera/status", methods=["GET"])
def camera_status():
    return jsonify({"is_running": True, "metrics": {}})

@app.route("/api/scene/analyze", methods=["POST"])
def analyze_scene():
    payload = request.get_json(silent=True) or {}
    image_base64 = payload.get("image")
    speak_flag = bool(payload.get("speak"))
    
    metrics = {}
    if not image_base64:
        return jsonify({"status": "error", "message": "No image provided"}), 400
        
    frame = base64_to_cv2(image_base64)
    analysis = call_openai_vision_api(frame, metrics=metrics)
    
    return jsonify({"analysis": analysis, "spoken": speak_flag, "metrics": metrics})


# ==== ROUTES: VOICE ==========================================================

@app.route("/api/voice/process", methods=["POST"])
def process_voice():
    pipeline_start = time.perf_counter()
    metrics = {}

    audio_file = request.files.get("audio")
    mode = request.form.get("mode", "plain")
    image_base64 = request.form.get("image")

    if not audio_file:
        return jsonify({"status": "error", "message": "No audio provided"}), 400

    # Preserve the original file extension (e.g., .webm or .mp4) sent by the browser
    ext = os.path.splitext(audio_file.filename)[1]
    if not ext:
        ext = ".webm"
        
    temp_path = f"temp_upload{ext}"
    audio_file.save(temp_path)

    # 1. Transcription (Whisper API)
    try:
        transcribe_start = time.perf_counter()
        with open(temp_path, "rb") as f:
            transcript_res = client_ai.audio.transcriptions.create(model="whisper-1", file=f)
        user_text = transcript_res.text
        metrics["transcription_ms"] = _elapsed_ms(transcribe_start)
    except Exception as e:
        print(e)
        return jsonify({"status": "error", "message": f"Transcription failed: {e}"}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    if not user_text:
         return jsonify({"status": "success", "message": "No speech detected"}), 200

    print(f"User said: {user_text}")

    # 2. Get AI Response Streaming
    llm_start = time.perf_counter()
    if mode == "scene" and image_base64:
        frame = base64_to_cv2(image_base64)
        llm_generator = get_ai_response_with_vision_streaming(user_text, frame)
    else:
        llm_generator = get_ai_response_streaming(user_text)

    def generate_sse():
        # Yield the transcript first
        yield f"data: {json.dumps({'type': 'transcript', 'text': user_text})}\n\n"

        # Queue to pass (text_chunk, tts_future) from background thread to main generator
        out_queue = queue.Queue()
        # Thread pool to fetch TTS concurrently
        tts_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

        def fetch_tts(text_chunk):
            """Runs in background thread pool to fetch audio"""
            try:
                audio_response = client_ai.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input=text_chunk,
                    response_format="mp3"
                )
                return base64.b64encode(audio_response.content).decode('utf-8')
            except Exception as e:
                print(f"TTS Error on chunk: {e}")
                return None

        def llm_worker():
            """Runs in a background thread to consume the LLM stream without blocking"""
            buffer = ""
            full_response_parts = []
            
            for chunk in llm_generator:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue
                
                full_response_parts.append(delta)
                buffer += delta
                
                parts = re.split(r'(?<=[.?!,])\s+', buffer)
                if len(parts) > 1:
                    for part in parts[:-1]:
                        text_chunk = part.strip()
                        if text_chunk:
                            # Start TTS generation IMMEDIATELY in parallel
                            future = tts_executor.submit(fetch_tts, text_chunk)
                            out_queue.put(("chunk", text_chunk, future))
                    
                    buffer = parts[-1]

            # Flush remaining buffer
            if buffer.strip():
                text_chunk = buffer.strip()
                future = tts_executor.submit(fetch_tts, text_chunk)
                out_queue.put(("chunk", text_chunk, future))

            # Signal completion
            out_queue.put(("done", "".join(full_response_parts), None))

        # Start LLM processing in the background
        threading.Thread(target=llm_worker).start()

        # Main thread consumes the queue and yields SSEs in order
        while True:
            msg_type, content, future = out_queue.get()
            
            if msg_type == "done":
                metrics["pipeline_total_ms"] = _elapsed_ms(pipeline_start)
                yield f"data: {json.dumps({'type': 'done', 'metrics': metrics, 'full_response': content})}\n\n"
                break
            
            # 1. Yield text immediately so frontend can display it instantly
            yield f"data: {json.dumps({'type': 'text', 'text': content})}\n\n"
            
            # 2. Wait for the TTS generation for THIS specific chunk to finish
            if future:
                audio_b64 = future.result() 
                if audio_b64:
                    yield f"data: {json.dumps({'type': 'audio', 'audio': audio_b64})}\n\n"

        tts_executor.shutdown(wait=False)

    return Response(stream_with_context(generate_sse()), mimetype='text/event-stream')

@app.route("/api/voice/stop-speech", methods=["POST"])
def api_stop_speech():
    # Because speech is played on the frontend now, backend has nothing to stop
    return jsonify({
        "status": "success",
        "message": "Speech playback stopped on frontend.",
        "metrics": {"operation_ms": 0.0},
    })


@app.route("/api/memory/capture", methods=["POST"])
def manual_capture():
    payload = request.get_json(silent=True) or {}
    image_base64 = payload.get("image")
    if not image_base64:
        return jsonify({"status": "error", "message": "No image provided"}), 400
        
    return jsonify(capture_memory_snapshot(image_base64))


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True, host='0.0.0.0')