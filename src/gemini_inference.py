"""
FastVLM inference wrapper.

Uses local FastVLM (LLaVA-based) code to generate
text descriptions from camera frames.
"""

# ------------------------------------------------------------------
# CRITICAL: make fastvlm/ visible to Python BEFORE importing llava
# ------------------------------------------------------------------
import os  # Import the os module for operating system dependent functionality
import sys  # Import the sys module to access system-specific parameters and functions

SRC_DIR = os.path.dirname(__file__)  # Get the directory name where the current script is located
REPO_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))  # Calculate the absolute path to the repository root by going one level up

FASTVLM_DIR = os.path.join(REPO_ROOT, "fastvlm")  # Construct the path to the 'fastvlm' directory within the repository root

# ONLY add fastvlm/, NOT fastvlm/llava
if FASTVLM_DIR not in sys.path:  # Check if the fastvlm directory is not already in the system path
    sys.path.insert(0, FASTVLM_DIR)  # Insert the fastvlm directory at the beginning of the system path to ensure it's found

print("[FastVLM] sys.path includes:", FASTVLM_DIR)  # Print a message confirming that the fastvlm directory has been added to the path

# ------------------------------------------------------------------
# Now imports will work
# ------------------------------------------------------------------
import torch  # Import the torch library for PyTorch functionality
import numpy as np  # Import the numpy library for numerical operations, aliased as np
from PIL import Image  # Import the Image class from the Pillow library for image processing

from llava.utils import disable_torch_init  # Import the disable_torch_init function to prevent unnecessary initialization overhead
from llava.conversation import conv_templates  # Import conversation templates for structuring the model's input
from llava.model.builder import load_pretrained_model  # Import the function to load the pre-trained LLaVA model
from llava.mm_utils import (  # Import multi-modal utility functions from LLaVA
    tokenizer_image_token,  # Helper to tokenise images
    process_images,  # Helper to process images for the model
    get_model_name_from_path,  # Helper to extract the model name from a file path
)
from llava.constants import (  # Import constant values used by the LLaVA model
    IMAGE_TOKEN_INDEX,  # Index used for image tokens
    DEFAULT_IMAGE_TOKEN,  # The default string token for images
    DEFAULT_IM_START_TOKEN,  # The token marking the start of an image
    DEFAULT_IM_END_TOKEN,  # The token marking the end of an image
)

# ------------------------------------------------------------------
# Singleton model state (loaded once)
# ------------------------------------------------------------------
_tokenizer = None  # Global variable to store the tokenizer instance, initialized to None
_model = None  # Global variable to store the model instance, initialized to None
_image_processor = None  # Global variable to store the image processor instance, initialized to None
_device = None  # Global variable to store the device (CPU/GPU) information, initialized to None


def _load_fastvlm():  # Define an internal function to load the FastVLM model components
    global _tokenizer, _model, _image_processor, _device  # Declare that we are using the global variables for the model components

    if _model is not None:  # Check if the model is already loaded
        return  # If the model is loaded, return immediately to avoid reloading

    disable_torch_init()  # Disable standard PyTorch initialization to speed up loading (often used in inference)

    # VERIFY THIS DIRECTORY EXISTS
    model_path = os.path.join(  # Construct the full path to the specific FastVLM model directory
        FASTVLM_DIR,  # Base fastvlm directory
        "llava",  # Subdirectory for llava
        "llava-fastvithd_0.5b_stage3"  # Specific model folder name
    )

    if not os.path.isdir(model_path):  # Check if the constructed model path exists
        raise RuntimeError(  # Raise a runtime error if the model path is missing
            f"FastVLM model not found at: {model_path}\n"  # Error message with path
            "Did you run get_models.sh?"  # Suggestion to run the setup script
        )

    model_name = get_model_name_from_path(model_path)  # Extract the model name from the directory path

    if torch.cuda.is_available():  # Check if a CUDA-compatible GPU is available
        _device = "cuda"  # Set the device to CUDA
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # Check if Apple MPS (Metal Performance Shaders) is available
        _device = "mps"  # Set the device to MPS
    else:  # If neither CUDA nor MPS is available
        _device = "cpu"  # Fallback to using the CPU

    print(f"[FastVLM] Loading model '{model_name}' on {_device}")  # Log the model name and the device being used

    _tokenizer, _model, _image_processor, _ = load_pretrained_model(  # Load the model, tokenizer, and image processor
        model_path=model_path,  # Path to the model
        model_base=None,  # Base model path (None implies it's self-contained or default)
        model_name=model_name,  # Name of the model
        device=_device,  # Device to load the model onto
    )

    _model.eval()  # Set the model to evaluation mode (disables dropout, etc.)


def describe_frame(frame: np.ndarray) -> str:
    """
    Generate a textual description of an OpenCV camera frame using Gemini 2.0 Flash.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables.")
        return "Error: Missing API Key"

    try:
        from google import genai
        from google.genai import types
        import cv2
        import base64
        
        client = genai.Client(api_key=api_key)
        
        # Convert OpenCV frame (BGR) to JPG bytes
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # We can pass raw bytes directly to the new Gemini SDK by wrapping in a Part
        image_part = types.Part.from_bytes(
            data=buffer.tobytes(),
            mime_type='image/jpeg'
        )
        
        prompt_text = (
            "Provide a short one paragraph summary of the scene, followed by a brief list of "
            "key people and their clothing, and any specific items or text visible. "
            "Be objective and concise. Avoid describing walls, floors, or empty space."
        )
        
        # Use gemini-2.5-flash which is Google's extremely fast and capable multimodal model
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image_part, prompt_text],
            config=types.GenerateContentConfig(
                 max_output_tokens=1000,
                 temperature=0.4,
            )
        )
        
        return response.text.strip()
        
    except Exception as e:
        print(f"Error calling Gemini Vision API: {e}")
        return f"Error: {e}"
