# coding=iso-8859-1
from typing import Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pi_change_detector.change_detector import ChangeDetector
from pi_face_detector.face_detector import FaceDetector
from pi_face_recognizer.face_recognizer import FaceRecognizer
from pi_face_recognizer import database
from pi_recogntion_logger.recognition_logger import RecognitionLogger
import cv2
import time
import json
import asyncio
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
RECOGNIZE_MODEL_PATH = os.environ.get("FACE_RECOGNIZE_MODEL", "")
RECOGNIZE_DB = database.load_database(os.environ.get("FACE_RECOGNIZE_DB", ""))
DEBUG = os.environ.get("DEBUG", "0") == "1"
LOG_THRESHOLD_MINUTES = int(os.environ.get("LOG_THRESHOLD_MINUTES", "5"))
TIMEZONE_OFFSET_HOURS = int(os.environ.get("TIMEZONE_OFFSET_HOURS", "-3"))

connected_clients: Set[WebSocket] = set()

log_queue: asyncio.Queue = None

logger = RecognitionLogger(threshold_seconds=LOG_THRESHOLD_MINUTES * 60, timezone_offset_hours=TIMEZONE_OFFSET_HOURS)

@app.on_event("startup")
async def startup_event():
    """Capture the main event loop when the app starts."""
    global log_queue
    log_queue = asyncio.Queue()
    print("✅ Log queue initialized")
    
    # Start a background task that reads from the queue and broadcasts
    asyncio.create_task(broadcast_worker())
    print("✅ Broadcast worker started")

async def broadcast_worker():
    """
    Async task that runs in the main event loop.
    Reads log entries from the queue and broadcasts them to all clients.
    """
    global connected_clients
    
    print("🔄 Broadcast worker is now listening for log entries...")
    while True:
        try:
            # Wait for a log entry from the queue (blocks until one arrives)
            log_entry = await log_queue.get()
            
            if not connected_clients:
                print(f"⚠️ No clients connected. Dropping log for {log_entry.get('person_name')}")
                continue
            
            message = json.dumps(log_entry)
            print(f"📡 Broadcasting to {len(connected_clients)} clients: {log_entry.get('person_name')}")
            
            disconnected = set()
            for client in connected_clients:
                try:
                    await client.send_text(message)
                    print("✅ Sent to client")
                except Exception as e:
                    print(f"❌ Failed to send to client: {e}")
                    disconnected.add(client)
            
            connected_clients -= disconnected
            
        except Exception as e:
            print(f"❌ Error in broadcast worker: {e}")
    
def schedule_broadcast(log_entry: dict):
    """
    Thread-safe way to add a log entry to the queue from the sync worker thread.
    """
    if log_queue is None:
        print("⚠️ Log queue not ready yet, skipping broadcast")
        return
    
    # put_nowait is thread-safe for asyncio.Queue
    try:
        log_queue.put_nowait(log_entry)
        print(f"📥 Added to queue: {log_entry.get('person_name')} (Queue size: {log_queue.qsize()})")
    except Exception as e:
        print(f"❌ Failed to add to queue: {e}")

# --- WebSocket Endpoint ---
@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    print(f"🔌 Frontend connected. Total clients: {len(connected_clients)}")
    
    try:
        # Keep connection alive without blocking on receive
        while True:
            # Use receive with a timeout so it doesn't block forever
            try:
                # Wait for any message from client (with 1 second timeout)
                await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except asyncio.TimeoutError:
                # No message received, just continue the loop
                # This keeps the connection alive and allows sends to work
                continue
    except WebSocketDisconnect:
        print("🔌 Frontend disconnected.")
    except Exception as e:
        print(f"⚠️ WebSocket error: {e}")
    finally:
        connected_clients.discard(websocket)
        print(f"🔌 Client removed. Remaining clients: {len(connected_clients)}")

def generate_frames():
    change_detector = ChangeDetector(min_area=1000, diff_threshold=25)
    face_detector = FaceDetector(FACE_MODEL_PATH, confidence_threshold=0.6)
    face_recognizer = FaceRecognizer(RECOGNIZE_MODEL_PATH, RECOGNIZE_DB)
    
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
                if DEBUG:
                    # A significant change was detected!
                    print(f"[ALERT] Motion detected! {pct:.2f}% of the screen changed.")
                
                # Example Action: Save the image if at least 5 seconds have passed since the last save
                # (This prevents saving 30 images in one second when someone walks by)
                current_time = time.time()
                if current_time - last_save_time > 5:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    
                    if DEBUG:
                        cv2.imwrite(f'motion_detected_{timestamp}.jpg', frame)
                        print(f"Saved motion capture: motion_detected_{timestamp}.jpg")
                        
                    last_save_time = current_time
                    
                    # --- FACE RECOGNITION LOGIC ---
                    detections = face_detector.recognize(frame)
                    
                    if detections:
                        for detection in detections:
                            name, score = face_recognizer.recognize_face(detection['crop'])
                            detection['name'] = name
                            
                            if logger.should_log(name):
                                log_entry = logger.create_log_entry(
                                    person_name=name,
                                    confidence=detection['confidence'],
                                    crop_image=detection.get('crop')
                                )
                                
                                if DEBUG:
                                    print(f"📝 LOGGED: {name} at {log_entry['formatted_time']}")
                                
                                # Broadcast to all connected frontends (async)
                                # We use asyncio.create_task to not block the frame generation
                                schedule_broadcast(log_entry)
                            else: 
                                print(f"Not logging recognition of {name}")
                            
                    else:
                        print("No faces detected")

                
                # Temporarily replace the normal yield block with this to see the motion mask:
                # if delta is not None:
                #     # Convert the black/white delta frame back to color (so the browser can display it)
                #     delta_color = cv2.cvtColor(delta, cv2.COLOR_GRAY2BGR)
                #     ret, buffer = cv2.imencode('.jpg', delta_color)
                #     frame_bytes = buffer.tobytes()
                #     yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            if detections:                    
                if DEBUG:
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

@app.get("/health")
def health_check():
    return {"status": "ok", "connected_clients": len(connected_clients)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)