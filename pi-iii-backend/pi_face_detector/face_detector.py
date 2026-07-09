# face_recognizer.py
import cv2
import os
import numpy as np
from ultralytics import YOLO
from typing import List, Dict

class FaceDetector:    
    def __init__(self, model_path, confidence_threshold: float = 0.5):
        """
        Initializes the YOLO-based face recognizer.
        
        :param model_path: Path to your custom trained .pt model file
        :param confidence_threshold: Minimum confidence to consider a detection valid (0.0 to 1.0)
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Get class names from the model (these should be the names of people you trained)
        self.class_names = self.model.names
        print(f"✅ Loaded face recognition model with {len(self.class_names)} classes:")
        for idx, name in self.class_names.items():
            print(f"   [{idx}] {name}")

    def recognize(self, frame: np.ndarray) -> List[Dict]:
        """
        Runs face recognition on the given frame.
        
        :param frame: OpenCV image in BGR format
        :return: List of dictionaries containing detection info:
                 [
                     {
                         'name': 'person_name',
                         'confidence': 0.95,
                         'bbox': (x1, y1, x2, y2),
                         'center': (cx, cy)
                     },
                     ...
                 ]
        """
        results = []
        
        # Run inference (verbose=False prevents printing to console every frame)
        predictions = self.model(frame, verbose=False, conf=self.confidence_threshold)
        
        for prediction in predictions:
            boxes = prediction.boxes
            
            for box in boxes:
                # Extract bounding box coordinates (xyxy format: x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Extract confidence score
                confidence = float(box.conf[0])
                        
                # Calculate center point (useful for tracking)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                face_crop = frame[y1:y2, x1:x2].copy()
                
                results.append({
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'crop': face_crop
                })
        
        return results

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draws bounding boxes and labels on the frame for visualization.
        
        :param frame: Original OpenCV image
        :param detections: List of detection dictionaries from recognize()
        :return: Annotated frame
        """
        annotated_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            
            # Draw bounding box (green)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare label text
            label = f"({confidence:.2f})"
            
            # Draw label background (filled rectangle)
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(
                annotated_frame, 
                (x1, y1 - label_size[1] - 10), 
                (x1 + label_size[0], y1), 
                (0, 255, 0), 
                -1
            )
            
            # Draw label text (black text on green background)
            cv2.putText(
                annotated_frame, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 0, 0), 
                2
            )
        
        return annotated_frame