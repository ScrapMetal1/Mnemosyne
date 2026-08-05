# Omni AR

Omni AR is an MDN project with the goal of creating an AI assistant with perfect memory recall that eventually could be deployed onto smart glasses / AR Glasses for general users and people with disabilities. It features a React-based frontend for real-time webcam capture and voice assistant features, connected to a Python backend handling vision analysis, text-to-speech, and memory extraction.

---

## Quick start

1. **Clone** the repo

   ```bash
   git clone https://github.com/ScrapMetal1/Omni AR.git
   cd Omni AR
   ```
2. **Set up environment variables** (see [API Keys Setup](#api-keys-setup))
3. **Create the conda environment** (see [Environment Setup](#environment-setup))
4. **Run the backend server**

   ```bash
   conda activate mdn_ar
   python src/service.py
   ```

5. **Run the React frontend**

   Open a new terminal window:
   ```bash
   cd react_ui
   npm install
   npm run dev
   ```

   Open your browser to the local URL provided by Vite (e.g., `http://localhost:5173`).

---

## Requirements

* Python 3.11 (works on 3.10–3.12)
* Conda (Anaconda or Miniconda) - **must be added to PATH**
* Webcam (internal or USB)
* API keys for OpenAI and ElevenLabs (see [API Keys Setup](#api-keys-setup))
* Node.js 16+ and npm (required for the React UI)

---

## Environment Setup

### Prerequisites: Conda Installation and PATH Configuration

**Important:** Ensure Conda is properly installed and added to your system PATH.

#### For Anaconda Users:
1. Install [Anaconda](https://www.anaconda.com/products/distributor) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. **During installation, check "Add Anaconda to my PATH environment variable"**
3. If you missed this step, add these to your PATH:
   - Windows: `C:\Users\<username>\anaconda3\Scripts\` and `C:\Users\<username>\anaconda3\`
   - macOS/Linux: `~/anaconda3/bin/` or `~/miniconda3/bin/`

#### Verify Conda Installation:
```bash
conda --version
conda info
```

If conda commands don't work, restart your terminal/command prompt or run the appropriate activation script.

### Option A — Anaconda Navigator (GUI)

**Goal:** set up the `mdn_ar` environment via the GUI.

1. Open **Anaconda Navigator → Environments → Import**
2. Choose **`environment.yml`** from the repo root
3. Name it **`mdn_ar`** and click **Import**
4. When it completes, click the **play ▶** icon → **Open Terminal**
5. Start the backend: `python src/service.py`

### Option B — Conda CLI (Recommended for Team Members)

From the repo root:

#### For Systems WITH NVIDIA GPU:
```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate mdn_ar

# Install CUDA-enabled PyTorch (if not automatically detected)
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### For Systems WITHOUT NVIDIA GPU:
```bash
# Create environment (PyTorch will install CPU-only automatically)
conda env create -f environment.yml

# Activate environment
conda activate mdn_ar
```

#### Verification:
Test your installation:
```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

**Expected output:**
- **With GPU**: `PyTorch version: 2.5.1+cu121`, `CUDA available: True`
- **Without GPU**: `PyTorch version: 2.9.0+cpu`, `CUDA available: False`

---

## API Keys Setup

This project requires API keys for OpenAI and ElevenLabs services. Create a `.env` file in the project root:

### 1. Create the .env file:
```bash
# Create .env file in project root
touch .env  # Linux/macOS
# or create .env manually on Windows
```

### 2. Add your API keys:
```bash
# Open .env in your text editor and add:
OPENAI_API_KEY=your_openai_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

### 3. Get API Keys:

#### OpenAI API Key:
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up/Login to your account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key and add it to your `.env` file

#### ElevenLabs API Key:
1. Go to [ElevenLabs](https://elevenlabs.io/)
2. Sign up/Login to your account
3. Go to Profile → API Keys
4. Generate a new API key
5. Copy the key and add it to your `.env` file

### 4. Security Notes:
- **Never commit `.env` files** to version control
- **Never share API keys** publicly
- Each team member should use their own API keys
- Monitor your API usage on the respective platforms to avoid unexpected charges

---

## Run the App

You need to run both the Python backend and the React frontend simultaneously in separate terminals.

**Terminal 1 (Backend):**
```bash
conda activate mdn_ar
python src/service.py
```
This will start the Flask server on `http://localhost:5000`.

**Terminal 2 (Frontend):**
```bash
cd react_ui
npm run dev
```
Navigate to the local URL (usually `http://localhost:5173`) in your web browser. Ensure you grant camera and microphone permissions when prompted.

---

## What You Should See

In your browser, you will see the **MDN Assist** interface:
* A live video feed from your webcam (if the camera is started)
* Status chips indicating Camera, Microphone, and Speech status
* **Scene controls** to analyze the camera feed or save memories
* **Voice assistant** controls to talk to the AI, optionally with scene context
* An **AI scene description** panel showing the latest vision analysis
* A **Voice transcript & response** panel showing your speech and the AI's response

---

## Project Structure

```
Omni AR/
├── environment.yml          # Conda environment configuration
├── .env                     # Environment variables (API keys) - NOT committed
├── README.md                # This file
├── LICENSE
├── src/
│   ├── service.py           # Main Flask backend server
│   ├── fastvlm_inference.py # Vision/Language model processing
│   ├── embedding_storage.py # Memory embeddings and retrieval
│   ├── filtering.py         # Query filtering logic
│   ├── legacy_code/         # Older python-only scripts
│   └── testing_scripts/     # Various test scripts
└── react_ui/                # React frontend application
    ├── package.json
    ├── vite.config.js
    └── src/                 # React components, services, and assets
```

---

## How It Works

### Architecture
The application has been refactored into a client-server architecture:
1. **Frontend (React UI)**: Handles all local hardware interactions (webcam video capture and microphone audio streaming) within the browser.
2. **Backend (Python/Flask)**: Serves as the AI processing engine, handling LLM requests, vision APIs, and long-term memory operations.

### Core Features:
1. **Webcam Capture**: The React UI uses the browser's MediaDevices API to capture real-time frames from the webcam.
2. **Scene Analysis**: The frontend captures a frame as a base64 string and sends it to the backend (`service.py`), which uses the OpenAI Vision API (or local models like FastVLM) to analyze and describe the scene.
3. **Voice Interaction**: 
   - Audio is recorded in the browser and sent to the backend.
   - The backend transcribes the audio using OpenAI's Whisper API.
   - The transcribed text (along with optional image context) is sent to the LLM.
   - The backend streams the AI's text response back to the frontend using Server-Sent Events (SSE).
   - Text-to-Speech (TTS) chunks are generated and streamed back to the frontend for real-time playback.
4. **Memory Capture**: The system can capture snapshots, analyze them, generate embeddings, and store them locally for future recall.

---

## Common Issues

**Conda command not found:**
- Ensure conda is installed and added to PATH (see [Environment Setup](#environment-setup))
- Restart your terminal/command prompt

**API Key errors:**
- Ensure `.env` file exists in project root
- Check that API keys are correctly formatted in `.env`
- Verify keys are active on respective platforms
- Restart application after adding keys

**Webcam or Microphone not working:**
- Ensure you have granted permission in your web browser for the site to access your camera and microphone.
- Close other applications that might be using the camera.

**Backend not responding (CORS or Connection Error):**
- Ensure `python src/service.py` is actively running in a terminal.
- Verify the backend is listening on the expected port (usually `5000`).

---

## Development Workflow

* Branch naming: `feat/…`, `fix/…`, `docs/…`
* Make small, focused pull requests
* Ensure `.env` files are never committed (add to `.gitignore` if missing)

### PR Checklist

* [ ] Applications run locally (`service.py` and React UI)
* [ ] No secrets or large files committed (check `.env`, audio files, etc.)
* [ ] README updated if user-facing changes

---

## Roadmap

* [x] Basic OCR with webcam overlay (legacy)
* [x] Voice recognition and push-to-talk
* [x] AI chat integration (OpenAI)
* [x] Text-to-speech (ElevenLabs / OpenAI TTS)
* [x] React UI integration with Flask backend
* [x] Frontend handling of video capture
* [ ] Confidence filtering and text de-duplication
* [ ] Run-every-N-frames toggle for performance
* [ ] Offline translation capabilities
* [ ] Mobile deployment (Android ML Kit)
* [ ] AR glasses UX overlays (Unity/AR Foundation)
* [ ] Multi-language OCR support
* [ ] Voice command processing for OCR control

---

## Security & Privacy

* API keys are stored securely in local `.env` files only
* Never commit `.env` files or share API keys
* Camera feed is only captured and sent to the API when explicitly requested by the user
* Ensure you do not commit snapshots containing personal data or sensitive information.

---

## Licence

MIT

---

## Acknowledgements

* [OpenCV](https://opencv.org/) - Computer vision library
* [PyTorch](https://pytorch.org/) - Machine learning framework
* [OpenAI](https://openai.com/) - AI chat capabilities and TTS
* [React](https://reactjs.org/) & [Vite](https://vitejs.dev/) - Frontend framework

---

## This README is ai generated because why not?
