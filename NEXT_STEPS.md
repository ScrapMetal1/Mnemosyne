# Next Steps: Mnemosyne Photographic Memory System

## Current Status ✅

You have successfully implemented:
- **Front-end UI** (React) - Camera controls, voice interface, latency metrics
- **Backend Service** (`service.py`) - Flask API with camera, voice, and OpenAI Vision integration
- **Embedding Database** (`embedding_storage.py`) - Vector storage with search capabilities
- **FastVLM Model** - Available in Swift/iOS format, needs Python integration

## Critical Next Steps

### 1. **Integrate FastVLM for On-Device Captioning** 🔴 HIGH PRIORITY

**Current State:** Using OpenAI Vision API (cloud-based, not privacy-preserving)
**Goal:** Replace with FastVLM for local, on-device captioning

**Tasks:**
- [ ] Set up FastVLM Python inference pipeline
  - Load FastVLM model (checkpoint available in `fastvlm/checkpoints/`)
  - Create captioning function that processes camera frames
  - Generate text descriptions like: "10:00 AM: I am in a kitchen. There are silver keys on the wooden counter"
  
- [ ] Replace OpenAI Vision calls in `service.py`
  - Modify `call_openai_vision_api()` to use FastVLM instead
  - Keep the same function signature for compatibility
  - Add FastVLM model loading/initialization

**Files to modify:**
- `src/service.py` - Replace vision API calls
- Create `src/fastvlm_inference.py` - FastVLM model wrapper

**Resources:**
- FastVLM checkpoints: `fastvlm/checkpoints/llava-fastvithd_0.5b_stage3/`
- FastVLM Python code: `fastvlm/llava/` and `fastvlm/predict.py`

---

### 2. **Continuous Captioning Pipeline** 🔴 HIGH PRIORITY

**Goal:** Automatically capture and caption frames at intervals, storing them in the embedding database

**Tasks:**
- [ ] Create frame sampling service
  - Capture frames every N seconds (e.g., every 5-10 seconds)
  - Or use motion detection to trigger captures
  - Avoid processing every frame (power/thermal constraints)
  
- [ ] Integrate captioning → embedding pipeline
  - When frame is captured → FastVLM generates caption
  - Extract embedding from caption text
  - Store in `EmbeddingStore` with timestamp and metadata
  
- [ ] Add background processing thread
  - Run captioning in separate thread to avoid blocking camera feed
  - Queue frames for processing
  - Handle errors gracefully

**Files to create:**
- `src/captioning_service.py` - Continuous captioning pipeline
- Modify `src/service.py` - Add endpoint to start/stop continuous logging

**Example flow:**
```
Camera Frame → FastVLM Caption → EmbeddingExtractor → EmbeddingStore
```

---

### 3. **RAG Query System** 🔴 HIGH PRIORITY

**Goal:** Enable natural language queries like "Where did I leave my keys?" using semantic search

**Tasks:**
- [ ] Create query endpoint
  - Accept natural language question
  - Convert query to embedding using `EmbeddingExtractor`
  - Search `EmbeddingStore` for top-k similar memories
  
- [ ] Integrate LLM for answer generation
  - Use Qwen2-Instruct (or OpenAI as fallback) to process retrieved memories
  - Generate natural language answer from context
  - Return answer + relevant memories
  
- [ ] Add voice response
  - Use existing TTS pipeline to speak the answer

**Files to create:**
- `src/rag_service.py` - RAG query processing
- Modify `src/service.py` - Add `/api/memory/query` endpoint

**Example query flow:**
```
User: "Where did I leave my keys?"
  → Query embedding → Search database → Top 5 memories
  → LLM: "Based on your visual log, you left silver keys on the wooden kitchen counter at 10:00 AM today."
```

---

### 4. **Model Loading & Initialization** 🟡 MEDIUM PRIORITY

**Tasks:**
- [ ] FastVLM model loader
  - Load checkpoint from `fastvlm/checkpoints/`
  - Initialize tokenizer and model
  - Handle device placement (CPU/GPU/NPU if available)
  
- [ ] Embedding model initialization
  - Load Qwen2 tokenizer/model for embeddings
  - Or use FastVLM's language model for embeddings (more efficient)
  
- [ ] Startup sequence
  - Load models on Flask app startup
  - Add health check endpoint
  - Handle model loading errors gracefully

**Files to create:**
- `src/model_loader.py` - Centralized model loading
- Modify `src/service.py` - Initialize models at startup

---

### 5. **Database Integration & Persistence** 🟡 MEDIUM PRIORITY

**Current State:** `EmbeddingStore` saves to disk, but not integrated with service

**Tasks:**
- [ ] Initialize `EmbeddingStore` in Flask service
  - Create global instance on startup
  - Use persistent storage directory
  
- [ ] Add memory management endpoints
  - `/api/memory/count` - Get total memories
  - `/api/memory/clear` - Clear all memories
  - `/api/memory/search` - Direct search endpoint (for testing)
  
- [ ] Add metadata filtering
  - Time-based filtering (e.g., "last hour", "today")
  - Location tags (if GPS available)
  - Object tags (extracted from captions)

**Files to modify:**
- `src/service.py` - Add memory endpoints
- `src/embedding_storage.py` - Enhance metadata filtering (already has TODO)

---

### 6. **Performance Optimization** 🟡 MEDIUM PRIORITY

**Tasks:**
- [ ] Frame downsampling
  - Process lower resolution frames for captioning
  - Keep full resolution for display only
  
- [ ] Batch processing
  - Process multiple frames in batch if possible
  - Queue management for captioning requests
  
- [ ] Caching
  - Cache embeddings for identical frames
  - Avoid re-processing static scenes
  
- [ ] NPU/GPU acceleration
  - Detect available hardware
  - Use CoreML/MLX for Apple devices
  - Use CUDA for NVIDIA GPUs

---

### 7. **Testing & Validation** 🟢 LOW PRIORITY

**Tasks:**
- [ ] Unit tests for embedding storage
- [ ] Integration tests for captioning pipeline
- [ ] End-to-end test: capture → caption → store → query
- [ ] Performance benchmarks
  - Captioning latency
  - Search latency
  - Memory usage

---

## Implementation Order (Recommended)

1. **Week 1: FastVLM Integration**
   - Set up FastVLM Python inference
   - Replace OpenAI Vision API
   - Test captioning on sample frames

2. **Week 2: Continuous Captioning**
   - Build frame sampling service
   - Integrate captioning → embedding pipeline
   - Test continuous logging

3. **Week 3: RAG Query System**
   - Build query endpoint
   - Integrate LLM for answer generation
   - Test end-to-end queries

4. **Week 4: Polish & Optimization**
   - Performance tuning
   - Error handling
   - UI integration for memory queries

---

## Quick Start: FastVLM Integration

Here's a minimal example to get started:

```python
# src/fastvlm_inference.py
import torch
from llava.model import *
from llava.mm_utils import *

def load_fastvlm_model(checkpoint_path):
    """Load FastVLM model from checkpoint"""
    # Implementation based on fastvlm/predict.py
    pass

def caption_frame(model, tokenizer, image, prompt="Describe this scene concisely."):
    """Generate caption for a single frame"""
    # Process image → tokens → caption
    pass
```

Then in `service.py`:
```python
from src.fastvlm_inference import load_fastvlm_model, caption_frame

# On startup
fastvlm_model, fastvlm_tokenizer = load_fastvlm_model("fastvlm/checkpoints/...")

# Replace OpenAI call
def call_fastvlm_api(frame, user_question=None):
    caption = caption_frame(fastvlm_model, fastvlm_tokenizer, frame)
    return caption
```

---

## Questions to Consider

1. **Frame Rate:** How often should frames be captured? (Every 5s? 10s? Motion-based?)
2. **Storage Limits:** How many memories to keep? (Time-based expiration? Size limits?)
3. **Privacy:** Should raw frames be discarded immediately? (Yes, per your architecture)
4. **Hardware:** Target platform? (Desktop for now? Apple Silicon? Mobile later?)
5. **LLM Choice:** Use Qwen2-Instruct locally, or keep OpenAI as fallback?

---

## Files Reference

- **Backend:** `src/service.py` - Main Flask API
- **Database:** `src/embedding_storage.py` - Vector storage
- **Frontend:** `react_ui/src/App.jsx` - React UI
- **FastVLM:** `fastvlm/` - Model and inference code
- **Voice:** `src/voice_reg.py` - Voice recognition (standalone)

---

Good luck! Start with FastVLM integration - that's the foundation for everything else.




