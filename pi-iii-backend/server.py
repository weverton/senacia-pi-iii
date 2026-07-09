# coding=iso-8859-1
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pi_change_detector.change_detector import ChangeDetector
from pi_face_detector.face_detector import FaceDetector
import cv2
import time
from dotenv import load_dotenv 
import os

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Allow Vue frontend to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change to your Vue URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

RTSP_URL = os.environ.get("CAMERA_URL", "")
FACE_MODEL_PATH = os.environ.get("FACE_DETECTION_MODEL", "")

def generate_frames():
    change_detector = ChangeDetector(min_area=1000, diff_threshold=25)
    face_detector = FaceDetector(FACE_MODEL_PATH, confidence_threshold=0.6)
    
    last_save_time = 0
    
    detections = None
    
    # Initialize the camera capture
    cap = cv2.VideoCapture(RTSP_URL)
    
    # CRITICAL: Set buffer size to 1 to prevent latency buildup 
    # (otherwise, the video will lag further and further behind real-time)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 

    try:
        while True:
            success, frame = cap.read()
            if not success:
                # If the camera disconnects, wait and retry
                time.sleep(1)
                cap = cv2.VideoCapture(RTSP_URL)
                continue
            
            is_changed, pct, delta = change_detector.compare(frame)
            
            if is_changed:
                # A significant change was detected!
                print(f"[ALERT] Motion detected! {pct:.2f}% of the screen changed.")
                
                # Example Action: Save the image if at least 5 seconds have passed since the last save
                # (This prevents saving 30 images in one second when someone walks by)
                current_time = time.time()
                if current_time - last_save_time > 5:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    cv2.imwrite(f'motion_detected_{timestamp}.jpg', frame)
                    print(f"Saved motion capture: motion_detected_{timestamp}.jpg")
                    last_save_time = current_time
                    
                    # --- FACE RECOGNITION LOGIC ---
                    detections = face_detector.recognize(frame)
                    
                    if not detections:                    
                        print("No faces detected")

                
                # Temporarily replace the normal yield block with this to see the motion mask:
                # if delta is not None:
                #     # Convert the black/white delta frame back to color (so the browser can display it)
                #     delta_color = cv2.cvtColor(delta, cv2.COLOR_GRAY2BGR)
                #     ret, buffer = cv2.imencode('.jpg', delta_color)
                #     frame_bytes = buffer.tobytes()
                #     yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            if detections:                    
                for det in detections:
                    print(f"👤 Recognized: ({det['confidence']:.2%} confidence)")
                
                # Draw bounding boxes and names on the frame
                frame = face_detector.draw_detections(frame, detections)
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            
            # Yield the frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()

@app.get("/video_feed")
def video_feed():
    # multipart/x-mixed-replace is the magic MIME type for MJPEG streams
    return StreamingResponse(
        generate_frames(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)