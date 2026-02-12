import cv2
print("cv2 imported successfully!")
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
         print("Error: Could not open camera.")
    else:
         print("cv2.VideoCapture available and opened!")
         cap.release()
except Exception as e:
    print(f"Error accessing camera: {e}")
