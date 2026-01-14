import numpy
import cv2
import sys
import os


from fastvlm_inference import describe_frame

def test_inference():
    # Create a dummy image (black square 512x512)
    # FastVLM expects a numpy array (OpenCV style: BGR)
    print("Creating dummy image...")
    dummy_frame = numpy.zeros((512, 512, 3), dtype=numpy.uint8)
    
    # Draw a white rectangle so there's something to describe
    cv2.rectangle(dummy_frame, (100, 100), (400, 400), (255, 255, 255), -1)
    
    print("Running describe_frame...")
    try:
        description = describe_frame(dummy_frame)
        print("\nSUCCESS! Description generated:")
        print("-" * 40)
        print(description)
        print("-" * 40)
    except Exception as e:
        print(f"\nERROR: Failed to run inference.\n{e}")

if __name__ == "__main__":
    test_inference()
