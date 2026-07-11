# recognition_logger.py
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional
import base64
import cv2

class RecognitionLogger:
    def __init__(self, threshold_seconds: int = 300, timezone_offset_hours: int = 0):
        """
        Logs recognized people with a cooldown threshold per person.
        
        :param threshold_seconds: Minimum seconds between logs for the same person (default: 5 min)
        :param timezone_offset_hours: Hours offset from UTC (e.g., -3 for GMT-3)
        """
        self.threshold_seconds = threshold_seconds
        self.last_logged: Dict[str, float] = {}
        
        # Create timezone object
        self.timezone = timezone(timedelta(hours=timezone_offset_hours))
        print(f"🕐 Logger timezone set to: UTC{timezone_offset_hours:+d}")

    def should_log(self, person_name: str) -> bool:
        """Check if enough time has passed to log this person again."""
        if ((person_name or "Desconhecido") == "Desconhecido"):
            return False
        
        current_time = time.time()
        last_time = self.last_logged.get(person_name, 0)
        
        if (last_time == 0) or (current_time - last_time >= self.threshold_seconds):
            self.last_logged[person_name] = current_time
            return True
        
        return False

    def create_log_entry(self, person_name: str, confidence: float, 
                         crop_image: Optional[any] = None) -> dict:
        """
        Creates a formatted log entry.
        
        :param crop_image: Optional OpenCV image (BGR) of the cropped face
        :return: Dictionary with log data
        """
        print(f"Creating log entry for {person_name}...")
        
        now = datetime.now(self.timezone)
        
        entry = {
            "person_name": person_name,
            "confidence": round(confidence, 4),
            "timestamp": now.isoformat(),
            "formatted_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        print(entry)
        
        # Convert cropped face to base64 for frontend display
        if crop_image is not None and crop_image.size > 0:
            try:
                _, buffer = cv2.imencode('.jpg', crop_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                entry["face_image"] = f"data:image/jpeg;base64,{img_base64}"
                print (f"Created face_image for {person_name}")
            except Exception as e:
                print(f"⚠️ Failed to encode face image: {e}")
                entry["face_image"] = None
        else:
            entry["face_image"] = None
        
        return entry