import asyncio
import cv2
import ssl
import os
import time
import math
import urllib.request

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.mediastreams import MediaStreamError

# ============================================================================
# CONFIGURAÇÃO — SERVIDOR WEBRTC
# ============================================================================

ICE_SERVERS = [
    RTCIceServer(urls="stun:stun.l.google.com:19302"),
]

HOST = "0.0.0.0"
PORT = 8000
USE_HTTPS = False
SSL_CERT = "cert.pem"
SSL_KEY = "key.pem"

WINDOW_NAME = "Phone Camera - Detecção"
SHOW_VIDEO = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# CONFIGURAÇÃO — DETECÇÃO DE ROSTO (OpenCV YuNet)
# ============================================================================

# Modelo YuNet (ONNX). Baixado automaticamente na primeira execução se não existir.
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_MODEL_PATH = os.path.join(BASE_DIR, "face_detection_yunet_2023mar.onnx")

CONFIDENCE = 0.6         # score_threshold do YuNet
NMS_THRESHOLD = 0.3
TOP_K = 50               # máximo de rostos por frame

# Diretório para salvar os recortes de rosto (dado sensível — restrinja o acesso)
SAVE_DIR = "faces"
SAVE_FACES = True
SAVE_COOLDOWN_SECONDS = 5

# Filtros de tamanho mínimo da caixa detectada (evita salvar rostos minúsculos/ruins)
MIN_FACE_WIDTH = 40
MIN_FACE_HEIGHT = 40

# Limita a taxa de inferência (frames de vídeo continuam fluindo mais rápido,
# mas o modelo só roda a cada 1/MAX_INFERENCE_FPS segundos)
MAX_INFERENCE_FPS = 15

# Tracker simples por centróide: distância máxima (px) para considerar que uma
# detecção é o mesmo rosto do frame anterior, e nº de frames sem detecção até
# "esquecer" o rosto
TRACK_MAX_DISTANCE = 80
TRACK_MAX_MISSES = 15

pcs = set()  # conexões WebRTC ativas


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def variance_of_laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def is_blurry(face_crop, threshold=80):
    return variance_of_laplacian(face_crop) < threshold


def save_face(face_crop, track_id, confidence):
    ensure_dir(SAVE_DIR)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SAVE_DIR, f"face_id{track_id}_conf{confidence:.2f}_{timestamp}.jpg")
    cv2.imwrite(filename, face_crop)
    return filename


def download_yunet_model():
    if os.path.exists(YUNET_MODEL_PATH):
        return
    print(f"⬇️  Baixando modelo YuNet para {YUNET_MODEL_PATH} ...")
    try:
        urllib.request.urlretrieve(YUNET_MODEL_URL, YUNET_MODEL_PATH)
        print("✅ Modelo baixado com sucesso!")
    except Exception as e:
        raise RuntimeError(
            f"Não foi possível baixar o modelo YuNet automaticamente ({e}). "
            f"Baixe manualmente de {YUNET_MODEL_URL} e salve como {YUNET_MODEL_PATH}"
        )


def load_model():
    print("📦 Carregando detector de rosto (OpenCV YuNet)...")
    try:
        download_yunet_model()
        model = cv2.FaceDetectorYN.create(
            YUNET_MODEL_PATH,
            "",
            (320, 320),  # tamanho de entrada inicial, ajustado por frame depois
            score_threshold=CONFIDENCE,
            nms_threshold=NMS_THRESHOLD,
            top_k=TOP_K,
        )
        print("✅ Modelo carregado com sucesso!")
        return model
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        raise


# ============================================================================
# TRACKER SIMPLES POR CENTRÓIDE
# O YuNet detecta rosto por rosto a cada frame, mas não dá um ID persistente
# como o ByteTrack. Esse tracker mínimo casa detecções entre frames pela
# distância do centro da caixa, só para termos um "ID" estável — útil para o
# cooldown de salvamento e, mais pra frente, para agrupar recortes por pessoa
# antes do reconhecimento facial.
# ============================================================================

class CentroidTracker:
    def __init__(self, max_distance=TRACK_MAX_DISTANCE, max_misses=TRACK_MAX_MISSES):
        self.max_distance = max_distance
        self.max_misses = max_misses
        self.next_id = 0
        self.tracks = {}  # id -> {"centroid": (x, y), "misses": int}

    def update(self, detections_centroids):
        """Recebe lista de centróides do frame atual, retorna lista de IDs na mesma ordem."""
        assigned_ids = [None] * len(detections_centroids)
        used_track_ids = set()

        for i, (cx, cy) in enumerate(detections_centroids):
            best_id, best_dist = None, self.max_distance
            for tid, t in self.tracks.items():
                if tid in used_track_ids:
                    continue
                tx, ty = t["centroid"]
                dist = math.hypot(cx - tx, cy - ty)
                if dist < best_dist:
                    best_dist, best_id = dist, tid

            if best_id is not None:
                self.tracks[best_id] = {"centroid": (cx, cy), "misses": 0}
                assigned_ids[i] = best_id
                used_track_ids.add(best_id)
            else:
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = {"centroid": (cx, cy), "misses": 0}
                assigned_ids[i] = new_id
                used_track_ids.add(new_id)

        # Envelhece tracks que não deram match neste frame
        for tid in list(self.tracks.keys()):
            if tid not in used_track_ids:
                self.tracks[tid]["misses"] += 1
                if self.tracks[tid]["misses"] > self.max_misses:
                    del self.tracks[tid]

        return assigned_ids


# ============================================================================
# DETECTOR (roda em cima de cada frame recebido via WebRTC)
# ============================================================================

class FaceDetector:
    def __init__(self, model):
        self.model = model
        self.tracker = CentroidTracker()
        self.frame_count = 0
        self.faces_detected_total = 0
        self.faces_saved_total = 0
        self.last_save_time = {}
        self.last_inference_time = 0
        self.cached_boxes = []  # lista de (x1, y1, x2, y2, conf)
        self.last_input_size = None
        self.fps = 0
        self.fps_counter = 0
        self.fps_timer = time.time()

    def process_frame(self, frame):
        """Processa um frame e retorna (frame_anotado, qtd_detectada_no_frame)."""
        self.frame_count += 1
        now = time.time()
        faces_in_frame = 0
        h, w = frame.shape[:2]

        run_inference = (now - self.last_inference_time) >= 1 / MAX_INFERENCE_FPS
        if run_inference:
            if self.last_input_size != (w, h):
                self.model.setInputSize((w, h))
                self.last_input_size = (w, h)

            _, faces = self.model.detect(frame)
            boxes = []
            if faces is not None:
                for f in faces:
                    x, y, fw, fh = f[0:4]
                    conf = f[14]
                    x1, y1 = int(x), int(y)
                    x2, y2 = int(x + fw), int(y + fh)
                    boxes.append((x1, y1, x2, y2, conf))
            self.cached_boxes = boxes
            self.last_inference_time = now

        boxes = self.cached_boxes

        # Filtra por tamanho mínimo antes de passar pro tracker
        valid_boxes = []
        for (x1, y1, x2, y2, conf) in boxes:
            if (x2 - x1) < MIN_FACE_WIDTH or (y2 - y1) < MIN_FACE_HEIGHT:
                continue
            valid_boxes.append((x1, y1, x2, y2, conf))

        centroids = [((x1 + x2) // 2, (y1 + y2) // 2) for (x1, y1, x2, y2, _) in valid_boxes]
        track_ids = self.tracker.update(centroids)

        for (x1, y1, x2, y2, conf), track_id in zip(valid_boxes, track_ids):
            face_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if face_crop.size == 0:
                continue

            faces_in_frame += 1
            self.faces_detected_total += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID {track_id} {conf:.2f}"
            cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if SAVE_FACES and not is_blurry(face_crop):
                last = self.last_save_time.get(track_id, 0)
                if now - last >= SAVE_COOLDOWN_SECONDS:
                    filename = save_face(face_crop, track_id, conf)
                    self.last_save_time[track_id] = now
                    self.faces_saved_total += 1
                    print(f"💾 Recorte salvo: {filename}")

        # FPS
        self.fps_counter += 1
        elapsed = time.time() - self.fps_timer
        if elapsed >= 1:
            self.fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_timer = time.time()

        frame = self._add_overlay(frame, faces_in_frame)
        return frame, faces_in_frame

    def _add_overlay(self, frame, faces_in_frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
        blended = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

        infos = [
            f"FPS: {self.fps:.1f}",
            f"Rostos no frame: {faces_in_frame}",
            f"Deteccoes totais: {self.faces_detected_total}",
            f"Recortes salvos: {self.faces_saved_total}",
            f"Frame: {self.frame_count}",
            "Modelo: OpenCV YuNet",
        ]
        for i, txt in enumerate(infos):
            cv2.putText(blended, txt, (10, 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return blended

    def get_stats(self):
        return {
            "frames": self.frame_count,
            "detections": self.faces_detected_total,
            "saved": self.faces_saved_total,
            "fps": self.fps,
        }


# Modelo e detector globais, carregados uma única vez na subida do servidor
model = load_model()
detector = FaceDetector(model)


# ============================================================================
# SERVIDOR WEBRTC
# ============================================================================

async def index(request):
    return web.FileResponse(os.path.join(BASE_DIR, "index.html"))


async def consume_video(track):
    """Recebe frames do celular via WebRTC e roda a detecção de rosto em cada um."""
    try:
        while True:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")

            # 🔥 processamento: detecção de rosto (MediaPipe) 🔥
            processed_img, _ = detector.process_frame(img)

            if SHOW_VIDEO:
                cv2.imshow(WINDOW_NAME, processed_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quitting...")
                break
            elif key == ord("s"):
                ensure_dir("snapshots")
                fname = f"snapshots/snapshot_{detector.frame_count}.jpg"
                cv2.imwrite(fname, processed_img)
                print(f"Snapshot salvo: {fname}")

            # cede o loop de eventos para não travar o asyncio
            await asyncio.sleep(0)

    except MediaStreamError:
        print("Video stream ended")
    except Exception as e:
        print(f"Error processing video: {e}")
    finally:
        cv2.destroyAllWindows()
        stats = detector.get_stats()
        print("📊 Resumo:", stats)
        print("Video processing stopped")


async def offer(request):
    params = await request.json()
    offer_desc = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ICE_SERVERS))
    pcs.add(pc)

    @pc.on("track")
    def on_track(track):
        print(f"Track received: {track.kind}")
        if track.kind == "video":
            asyncio.ensure_future(consume_video(track))

    @pc.on("connectionstatechange")
    def on_connectionstatechange():
        print(f"Connection state: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed", "disconnected"):
            pcs.discard(pc)

    await pc.setRemoteDescription(offer_desc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    })


async def health(request):
    return web.json_response({"status": "ok", "connections": len(pcs)})


async def on_shutdown(app):
    print("Shutting down...")
    await asyncio.gather(*(pc.close() for pc in pcs))
    pcs.clear()
    cv2.destroyAllWindows()


def create_app():
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/health", health)
    app.router.add_post("/offer", offer)
    return app


if __name__ == "__main__":
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║  Phone Camera → Python WebRTC Streamer + Face Detection       ║
    ║                                                                ║
    ║  Server running on: http://localhost:{PORT}                        ║
    ║  Health check: http://localhost:{PORT}/health                      ║
    ║                                                                ║
    ║   Rode: sudo tailscale serve --https 8443 localhost:8000     ║
    ║                                                                ║
    ║  Acesse pelo celular via:                                     ║
    ║    - Tailscale HTTPS: https://[machine].ts.net:8443          ║
    ║                                                                ║
    ║  Atalhos de teclado (janela do OpenCV):                       ║
    ║    q  → Sair                                                  ║
    ║    s  → Salvar snapshot do frame processado                   ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    app = create_app()

    if USE_HTTPS:
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(SSL_CERT, SSL_KEY)
        web.run_app(app, host=HOST, port=PORT, ssl_context=ssl_context)
    else:
        web.run_app(app, host=HOST, port=PORT)
