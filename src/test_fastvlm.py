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
    
    # Define test images
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGES_DIR = os.path.join(SCRIPT_DIR, "test_images")
    
    test_files = [
        "theatre.jpg",
        "macbookatdesk.jpg",
        "Keyoncounter.jpg",
        "Keyoncounter.png"
    ]

    for filename in test_files:
        print(f"\nRunning describe_frame with {filename}...")
        image_path = os.path.join(IMAGES_DIR, filename)
        
        # Check if file exists
        if not os.path.exists(image_path):
             print(f"ERROR: File not found at {image_path}")
             continue

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"ERROR: Failed to load image {filename}")
            continue

        try:
            description = describe_frame(img)
            print("SUCCESS! Description generated:")
            print("-" * 40)
            print(description)
            print("-" * 40)
        except Exception as e:
            print(f"ERROR: Failed to run inference on {filename}.\n{e}")

if __name__ == "__main__":
    test_inference()
