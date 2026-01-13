# Mnemosyne

Mnemosyne is a personal project with the goal of creating an AI assistant with perfect memory recall that eventually could be deployed onto smart glasses / AR Glasses for general users and people with disabilities. It currently includes a real-time webcam OCR prototype with voice assistant features, capturing frames from a camera, detecting and recognizing text on-device, and overlaying results.

---

## Quick start

1. **Clone** the repo

   ```bash
   git clone https://github.com/ScrapMetal1/Mnemosyne.git
   cd Mnemosyne
   ```
2. **Set up environment variables** (see [API Keys Setup](#api-keys-setup))
3. **Create the conda environment** (see [Environment Setup](#environment-setup))
4. **Run the applications**

   ```bash
   # Main OCR app
   python src/main.py

   # Voice assistant app
   python src/voice_reg.py

   # Merged app (OCR + voice features)
   python src/merged_functions.py
   ```

   Press `q` to quit, `s` to save a snapshot.

### React UI (Optional)

For the React frontend interface:

```bash
cd react_ui
npm install
npm run dev
```

This starts a development server with hot module replacement. The React UI provides an alternative interface to the Python applications.

---

## Requirements

* Python 3.11 (works on 3.10–3.12)
* Conda (Anaconda or Miniconda) - **must be added to PATH**
* Webcam (internal or USB)
* API keys for OpenAI and ElevenLabs (see [API Keys Setup](#api-keys-setup))
* Node.js 16+ and npm (optional, for React UI development)

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
5. Run one of the applications:

   ```bash
   python src/main.py          # Basic OCR
   python src/voice_reg.py     # Voice assistant
   python src/merged_functions.py    # Full featured app
   ```

If you do not see the environment, refresh Navigator or close and reopen it.

---

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

## Portable Build (Optional)

Create a zip you can copy to another machine with the **same** OS and architecture.

```bash
conda install -n mdn_ar -c conda-forge conda-pack
conda activate mdn_ar
conda pack -n mdn_ar -o mdn_ar_env.tar.gz
```

On the target machine:

```bash
mkdir mdn_ar_env
tar -xzf mdn_ar_env.tar.gz -C mdn_ar_env
# Linux/macOS
mdn_ar_env/bin/conda-unpack && source mdn_ar_env/bin/activate
# Windows (PowerShell)
mdn_ar_env\Scripts\conda-unpack.exe; mdn_ar_env\Scripts\activate
python src/main.py
```

To make Navigator see it, place the unpacked folder under your Conda `envs` directory or add its path to `~/.condarc` under `envs_dirs`.

---

## Environment Files

### `environment.yml`

```yaml
name: mdn_ar
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy
  - pip
  # Essential system dependencies only
  - ffmpeg
  - portaudio
  # All Python packages via pip for maximum compatibility
  - pip:
      # PyTorch - automatically detects CUDA availability
      - torch
      - torchvision
      - torchaudio
      # Other dependencies
      - opencv-python
      - easyocr
      - openai
      - sounddevice
      - soundfile
      - pynput
      - elevenlabs
      - SpeechRecognition
      - flask
      - flask-cors
      - python-dotenv
```

### Exact Lockfile (Optional)

Produce a bit‑for‑bit spec for this OS:

```bash
conda list --explicit > spec.txt
# recreate with
conda create -n mdn_ar --file spec.txt
```

---

## Run the Apps

```bash
conda activate mdn_ar

# Choose your application:
python src/main.py          # Basic OCR with webcam overlay
python src/voice_reg.py     # Push-to-talk voice assistant
python src/merged_functions.py    # Combined OCR + voice features
```

**Keys:** `q` quit • `s` save frame • `p` voice recording (in voice apps).
If the camera window is black, edit the respective `.py` file and try `VideoCapture(1)` or `VideoCapture(2)`.

---

## What You Should See

### OCR Applications (`main.py`, `merged_functions.py`):
* A window titled **"Mnemosyne - Webcam OCR"**
* Green polygons around detected text
* Label with recognised string and a confidence score (0–1)
* An FPS counter in the top‑left

### Voice Applications (`voice_reg.py`, `merged_functions.py`):
* Console output showing "Push-to-Talk Assistant Ready"
* Audio feedback when recording starts/stops
* AI responses played through speakers

Point the camera at a book page, a menu, or your monitor for OCR features.

---

## Project Structure

```
Mnemosyne/
├── environment.yml          # Conda environment configuration
├── .env                     # Environment variables (API keys) - NOT committed
├── README.md                # This file
├── LICENSE
├── src/
│   ├── main.py              # Basic OCR webcam application
│   ├── voice_reg.py         # Push-to-talk voice assistant
│   ├── merged_functions.py  # Combined OCR + voice features
│   ├── testing11labs.py     # ElevenLabs API testing script
│   ├── __pycache__/         # Python cache files (gitignored)
│   └── temp_recording.wav   # Temporary audio file (gitignored)
└── react_ui/                # React frontend (separate application)
    ├── package.json
    ├── vite.config.js
    ├── src/
    │   ├── App.jsx
    │   ├── main.jsx
    │   └── assets/
    └── public/
```

---

## How It Works

### OCR Applications (`main.py`, `merged_functions.py`):
1. OpenCV captures frames from the webcam in real-time
2. EasyOCR performs optical character recognition on each frame
3. Text detection results are overlaid as green polygons with confidence scores
4. FPS counter tracks performance in the top-left corner
5. Press 's' to save snapshots, 'q' to quit

### Voice Applications (`voice_reg.py`, `merged_functions.py`):
1. Continuous audio stream monitoring using sounddevice
2. Push-to-talk activation (hold 'p' key) or continuous listening modes
3. Speech recognition converts audio to text using Google's API
4. OpenAI processes the text for intelligent responses
5. ElevenLabs generates natural-sounding speech from AI responses
6. Audio playback through system speakers

### Combined Features (`merged_functions.py`):
- All OCR functionality plus voice interaction
- AI can describe what it sees in the camera feed
- Voice commands can control OCR behavior
- Integrated Flask server for potential web interface

You can tweak language packs, confidence thresholds, and OCR frequency to trade accuracy for speed.

---

## Common Issues

**Conda command not found:**
- Ensure conda is installed and added to PATH (see [Environment Setup](#environment-setup))
- Restart your terminal/command prompt
- On Windows, run `conda init` and restart

**Torch/EasyOCR install fails:**
- For CPU-only systems: The environment.yml should handle this automatically
- For GPU systems: Follow the CUDA installation steps in [Environment Setup](#environment-setup)
- Alternative manual install:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install easyocr
```

**API Key errors:**
- Ensure `.env` file exists in project root
- Check that API keys are correctly formatted in `.env`
- Verify keys are active on respective platforms
- Restart application after adding keys

**Webcam busy or black frame:**
- Close other camera-using applications
- Try different camera indices: `VideoCapture(1)` or `VideoCapture(2)`
- Check camera permissions on macOS/Linux

**Audio/microphone issues:**
- Ensure microphone permissions are granted
- Check audio device settings
- Try different audio devices if multiple are available
- On Windows, ensure correct audio drivers

**Low FPS:**
- Lower camera resolution in the source code
- Run OCR every N frames instead of every frame
- Downscale frames before OCR processing
- Use CPU-only PyTorch if GPU performance is poor

**CUDA not detected:**
- Install NVIDIA CUDA drivers from NVIDIA website
- Verify CUDA installation: `nvidia-smi`
- Reinstall PyTorch with CUDA support if needed

---

## Development Workflow

* Branch naming: `feat/…`, `fix/…`, `docs/…`
* Make small, focused pull requests
* Test all three applications before pushing: OCR, voice assistant, and merged app
* Ensure `.env` files are never committed (add to `.gitignore` if missing)

### Coding Style

* Keep functions small and pure where possible
* Prefer clear names over comments
* Add docstrings to all new functions
* Use type hints where beneficial
* Follow PEP 8 style guidelines

### Environment Management

* Update `environment.yml` when adding new dependencies
* Test environment creation on both GPU and CPU systems
* Document any special setup requirements in this README

### PR Checklist

* [ ] All applications run locally: `python src/main.py`, `python src/voice_reg.py`, `python src/merged_functions.py`
* [ ] No secrets or large files committed (check `.env`, audio files, etc.)
* [ ] README updated if user-facing changes
* [ ] Environment setup tested on clean conda environment
* [ ] API keys properly documented for new team members
* [ ] Screenshots/GIFs for UI changes (optional)

---

## Roadmap

* [x] Basic OCR with webcam overlay
* [x] Voice recognition and push-to-talk
* [x] AI chat integration (OpenAI)
* [x] Text-to-speech (ElevenLabs)
* [ ] Confidence filtering and text de-duplication
* [ ] Run-every-N-frames toggle for performance
* [ ] Offline translation capabilities
* [ ] React UI integration with Flask backend
* [ ] Mobile deployment (Android ML Kit)
* [ ] AR glasses UX overlays (Unity/AR Foundation)
* [ ] Multi-language OCR support
* [ ] Voice command processing for OCR control

---

## Security & Privacy

* All OCR processing happens on-device - no images are uploaded
* Voice data is processed locally before sending to APIs
* API keys are stored securely in local `.env` files only
* Never commit `.env` files or share API keys
* Audio recordings are temporary and deleted after processing
* Camera feed is not recorded unless explicitly saved with 's' key

**Important:** Do not commit snapshots containing personal data or sensitive information.

---

## Applications Overview

### `main.py` - Basic OCR
- Real-time webcam text detection
- Green polygon overlays on detected text
- Confidence scores and FPS display
- Simple snapshot saving

### `voice_reg.py` - Voice Assistant
- Push-to-talk voice interaction
- OpenAI-powered responses
- ElevenLabs text-to-speech
- Keyboard-activated recording

### `testing11labs.py` - ElevenLabs Testing
- Simple script to test ElevenLabs text-to-speech functionality
- Useful for verifying API key setup and audio output
- Plays a sample voice message when run

### `merged_functions.py` - Full Featured App
- Combines OCR and voice features
- AI can describe camera contents
- Voice commands for OCR control
- Flask server for potential web interface
- Most complete feature set

---

## Licence

MIT

---

## Acknowledgements

* [OpenCV](https://opencv.org/) - Computer vision library
* [EasyOCR](https://github.com/JaidedAI/EasyOCR) - Optical character recognition
* [PyTorch](https://pytorch.org/) - Machine learning framework
* [OpenAI](https://openai.com/) - AI chat capabilities
* [ElevenLabs](https://elevenlabs.io/) - Text-to-speech synthesis
* [Google Speech Recognition](https://pypi.org/project/SpeechRecognition/) - Voice recognition

---

## This README is ai generated because why not? 
