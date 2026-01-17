import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from embedding_storage import EmbeddingStore, EmbeddingExtractor
from fastvlm_inference import describe_frame, _load_fastvlm
import cv2
import time

#load the camera
cap = cv2.VideoCapture(0)

#load FastVLM weights
_load_fastvlm()

#instantiate the classes
extractor = EmbeddingExtractor()
store = EmbeddingStore()

# #capture the an image every 5 seconds. 
# for i in range(5):
#     print("Capturing Frame")        
#     ret, frame  = cap.read()
#     if not ret:
#         print("frame not captured")
 
#     #describe the frame
#     describe_start = time.time()
#     description = describe_frame(frame)
#     print(description)
#     describe_end = time.time()
#     print(f'FastVLM describe',describe_end-describe_start)



#     #embed the frame
#     embed_start = time.time()
#     embedding = extractor.extract_embeddings(description)
#     embed_end = time.time()
    
#     print(f"Embed Time:",embed_end-embed_start)

#     store.add(embedding=embedding, text=description)

    
#     print("sleeping")
#     time.sleep(5)





#search for memories

query = "What was my grandma wearing?"
query = "Where did I leave my tape?"


embedded_query = extractor.extract_embeddings(query)

print(store.search(embedded_query, 1))


    

    











