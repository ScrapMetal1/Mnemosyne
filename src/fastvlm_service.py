"""
FastVLM Service for On-Device Vision Captioning

This module provides FastVLM integration for generating text descriptions
of camera frames locally, without requiring cloud APIs.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from datetime import datetime
from typing import Optional, Tuple
import cv2

# Add fastvlm to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fastvlm'))

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


class FastVLMService:
    """
    Service for running FastVLM inference on camera frames.
    Provides on-device captioning without cloud dependencies.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize FastVLM service.
        
        Args:
            model_path: Path to FastVLM checkpoint directory. 
                       Defaults to fastvlm/checkpoints/llava-fastvithd_0.5b_stage3
            device: Device to run on ('cuda', 'mps', 'cpu'). Auto-detects if None.
        """
        if model_path is None:
            # Default to checkpoint in repo
            model_path = os.path.join(
                os.path.dirname(__file__), 
                '..', 
                'fastvlm', 
                'checkpoints', 
                'llava-fastvithd_0.5b_stage3'
            )
            model_path = os.path.abspath(model_path)
        
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.context_len = None
        self.device = device
        self._is_loaded = False
        
    def load_model(self):
        """Load FastVLM model and tokenizer."""
        if self._is_loaded:
            return
        
        print(f"[FastVLM] Loading model from {self.model_path}...")
        
        # Auto-detect device if not specified
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        
        print(f"[FastVLM] Using device: {self.device}")
        
        # Disable torch init for faster loading
        disable_torch_init()
        
        # Get model name from path
        model_name = get_model_name_from_path(self.model_path)
        
        # Load model
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            self.model_path, 
            model_base=None, 
            model_name=model_name, 
            device=self.device
        )
        
        # Set pad token for generation
        if hasattr(self.model, 'generation_config'):
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        
        self._is_loaded = True
        print("[FastVLM] Model loaded successfully!")
    
    def caption_frame(
        self, 
        frame: np.ndarray, 
        prompt: str = "Describe this scene concisely. Identify objects, people, text, and spatial layout.",
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        conv_mode: str = "qwen_2"
    ) -> str:
        """
        Generate caption for a camera frame.
        
        Args:
            frame: OpenCV frame (BGR numpy array)
            prompt: Prompt for the VLM
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            conv_mode: Conversation template mode
            
        Returns:
            Generated caption text
        """
        if not self._is_loaded:
            self.load_model()
        
        # Convert OpenCV BGR frame to PIL RGB Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Construct prompt with image token
        qs = prompt
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        
        # Build conversation
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
        
        # Tokenize prompt
        device_obj = torch.device(self.device)
        input_ids = tokenizer_image_token(
            prompt_text, 
            self.tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors='pt'
        ).unsqueeze(0).to(device_obj)
        
        # Process image
        image_tensor = process_images([pil_image], self.image_processor, self.model.config)[0]
        
        # Run inference
        with torch.inference_mode():
            # Prepare image tensor
            image_tensor_device = image_tensor.unsqueeze(0)
            if self.device == "cpu":
                image_tensor_device = image_tensor_device.float()
            else:
                image_tensor_device = image_tensor_device.half()
            image_tensor_device = image_tensor_device.to(device_obj)
            
            # Generate caption
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor_device,
                image_sizes=[pil_image.size],
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=None,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True
            )
            
            # Decode output
            caption = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            # Extract just the assistant's response (remove prompt)
            if conv.roles[1] in caption:
                caption = caption.split(conv.roles[1])[-1].strip()
            
            return caption
    
    def caption_with_timestamp(self, frame: np.ndarray, prompt: Optional[str] = None) -> str:
        """
        Generate timestamped caption in diary format.
        
        Example: "10:00 AM: I am in a kitchen. There are silver keys on the wooden counter."
        
        Args:
            frame: OpenCV frame
            prompt: Optional custom prompt
            
        Returns:
            Timestamped caption text
        """
        if prompt is None:
            prompt = "Describe this scene concisely. Identify objects, people, text, and spatial layout."
        
        caption = self.caption_frame(frame, prompt=prompt)
        
        # Add timestamp prefix
        timestamp = datetime.now().strftime("%I:%M %p")
        timestamped_caption = f"{timestamp}: {caption}"
        
        return timestamped_caption


# Global instance (lazy-loaded)
_fastvlm_service: Optional[FastVLMService] = None


def get_fastvlm_service(model_path: Optional[str] = None) -> FastVLMService:
    """Get or create global FastVLM service instance."""
    global _fastvlm_service
    if _fastvlm_service is None:
        _fastvlm_service = FastVLMService(model_path=model_path)
    return _fastvlm_service


# Example usage
if __name__ == "__main__":
    # Test with a sample image
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load image from {image_path}")
            sys.exit(1)
    else:
        print("Usage: python fastvlm_service.py <image_path>")
        print("Or use from service.py after integrating")
        sys.exit(0)
    
    # Initialize service
    service = FastVLMService()
    
    # Generate caption
    print("Generating caption...")
    caption = service.caption_with_timestamp(frame)
    print(f"\nCaption: {caption}")




