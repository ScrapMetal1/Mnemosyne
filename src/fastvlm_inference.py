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


def describe_frame(frame: np.ndarray) -> str:  # Define the public function to generate a description for a given image frame
    """
    Generate a textual description of an OpenCV camera frame.


    Note that the FastVLM processes a 1024x1024 image. That should be our target input.
    """
    _load_fastvlm()  # Ensure the model is loaded before processing

    # OpenCV BGR → PIL RGB
    image = Image.fromarray(frame[:, :, ::-1]).convert("RGB")  # Convert the OpenCV BGR numpy array to a PIL RGB image

    prompt_text = (  # Define the text prompt to guide the model's description
        "Provide a detailed one-sentence summary of the scene, with regard to "
        "key people and their clothing, and any specific items or text visible. "
        "Be objective and concise. Avoid describing walls, floors, or empty space."
    )

    qs = (  # Construct the query string including special image tokens
        DEFAULT_IM_START_TOKEN  # Start token for the image
        + DEFAULT_IMAGE_TOKEN  # Placeholder token where the image embedding will go
        + DEFAULT_IM_END_TOKEN  # End token for the image
        + "\n"  # Newline separator
        + prompt_text  # The actual text prompt
    )

    conv = conv_templates["qwen_2"].copy()  # Create a copy of the conversation template (using qwen_2 style)
    conv.append_message(conv.roles[0], qs)  # Add the user's message (query) to the conversation history
    conv.append_message(conv.roles[1], None)  # Add a placeholder for the assistant's response
    prompt = conv.get_prompt()  # Generate the full formatted prompt string

    device_obj = torch.device(_device)  # Create a torch.device object from the device string

    input_ids = tokenizer_image_token(  # Tokenize the text prompt specially handling the image token
        prompt,  # The full prompt string
        _tokenizer,  # The tokenizer to use
        IMAGE_TOKEN_INDEX,  # The index mapping for the image token
        return_tensors="pt",  # Return PyTorch tensors
    ).unsqueeze(0).to(device_obj)  # Add a batch dimension and move to the target device

    image_tensor = process_images(  # Process the image to get the tensor input for the model
        [image], _image_processor, _model.config  # Pass the image list, processor, and model config
    )[0].unsqueeze(0)  # Select the first image result (batch of 1) and add a batch dimension

    image_tensor = (  # Cast the image tensor to the appropriate data type
        image_tensor.float() if _device == "cpu" else image_tensor.half()  # Use float32 for CPU, float16 (half) for GPU/MPS
    ).to(device_obj)  # Move the tensor to the target device

    with torch.inference_mode():  # Disable gradient calculation for faster inference
        output_ids = _model.generate(  # Generate the output sequence from the model
            input_ids,  # The tokenized text input
            images=image_tensor,  # The processed image input
            image_sizes=[image.size],  # The size of the original image
            do_sample=False,  # deterministic generation
            temperature=0.7, # this doesn't matter if sample is set to false
            max_new_tokens=128,  # Maximum number of tokens to generate
            use_cache=True,  # Enable KV caching for speed
            repetition_penalty=1.1, # Help stop the model from looping on the same words

        )

    output_text = _tokenizer.batch_decode(
        output_ids, skip_special_tokens=True
    )[0].strip()

    # Clean up artifacts like <start>, <end>, or other special markers
    stop_words = [
        "<|im_end|>", "<|im_start|>", "<end>", "<start>",
        "<end of description>", "<start of description>",
        "<end of im_end>", "<im_end>", "<im_start>",
        "<end of response>", "<start of response>", 
        "Question:", "Answer:", "User:", "Assistant:", "###", "Context:", "\n"
    ]
    for stop in stop_words:
        if stop in output_text:
            output_text = output_text.split(stop)[0]

    # Handle <Instruction> / <Response> artifacts
    if "<Response:" in output_text:
        # Take everything after <Response: 
        output_text = output_text.split("<Response:")[1].strip()
        # Remove trailing '>' if it exists (though usually it's just text)
        if output_text.endswith(">"):
            output_text = output_text[:-1]

    return output_text.strip()
