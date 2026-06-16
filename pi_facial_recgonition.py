from ultralytics import YOLO
import cv2
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='./.pi-env')

debug = os.getenv("DEBUG") == "1"
model = YOLO(os.getenv("MODEL_PATH"))
results = model(os.getenv("FILES_PATH"), stream=True, conf=0.74)

i = 0
# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    
    if (boxes.shape[0]) > 0:
        img = np.array(result.orig_img, dtype=np.uint8)

        for index, box in enumerate(boxes):
            # Extract the xyxy coordinates as integers
            x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
            # NumPy slice the image array: img[ymin:ymax, xmin:xmax]
            cropped_img = img[y1:y2, x1:x2]
            
            # Save the isolated crop
            filename = f"result-{i}-crop{index}.jpg"
            cv2.imwrite(filename, cropped_img)
            if debug:
                print(f"Saved: {filename}")

        if debug:
            print(f'info on {i}:\nQuantidade:{boxes.shape[0]}\n{boxes}')
    else:
        print(f"No face(s) recognized on {i}")

    i += 1    
    if i > 5:
        break