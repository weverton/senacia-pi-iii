import cv2
import os
import time
import numpy as np
import requests
from ultralytics import YOLO
import threading
from queue import Queue

# =====================================================
# CONFIGURAÇÕES
# =====================================================

# Opções de conexão (tente diferentes URLs se uma não funcionar)
STREAM_OPTIONS = [
    "https://vdo.ninja/?view=vpHxLKi",  # Link original do VDO.ninja
    "http://localhost:8080/video",       # Para IP Webcam
    "http://192.168.1.100:8080/video",   # IP Webcam padrão
    "rtsp://192.168.1.100:8554/stream",  # RTSP stream
    0,                                    # Webcam local (0 = primeira câmera)
    1                                     # Webcam local (1 = segunda câmera)
]

# Modelo YOLO para detecção facial
# Usa o modelo padrão YOLO e filtra por pessoas (classe 0)
MODEL_NAME = "yolov8n.pt"  # Nano - mais leve
# MODEL_NAME = "yolov8s.pt"  # Small - um pouco melhor
# MODEL_NAME = "yolov8m.pt"  # Medium - melhor performance

CONFIDENCE = 0.55
WINDOW_NAME = "Detecção Facial - Stream"
DISPLAY_WIDTH = 1280

# Diretório para salvar faces
SAVE_DIR = "faces"
SAVE_FACES = True
SAVE_COOLDOWN_SECONDS = 5

# Filtros
MIN_FACE_WIDTH = 60
MIN_FACE_HEIGHT = 60

# Performance
MAX_INFERENCE_FPS = 15
RECONNECT_DELAY = 2

# Configuração VDO.ninja específica
VDO_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
VDO_HEADERS = {
    'User-Agent': VDO_USER_AGENT,
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Origin': 'https://vdo.ninja',
    'Referer': 'https://vdo.ninja/'
}

# =====================================================
# FUNÇÕES AUXILIARES
# =====================================================

def ensure_dir(path):
    """Cria diretório se não existir."""
    os.makedirs(path, exist_ok=True)

def variance_of_laplacian(image):
    """Calcula a variância do Laplaciano para detecção de borrão."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def is_blurry(face_crop, threshold=80):
    """Verifica se a imagem está borrada."""
    return variance_of_laplacian(face_crop) < threshold

def save_face(face_crop, track_id, confidence):
    """Salva o recorte facial com metadados."""
    ensure_dir(SAVE_DIR)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SAVE_DIR, f"face_id{track_id}_conf{confidence:.2f}_{timestamp}.jpg")
    cv2.imwrite(filename, face_crop)
    return filename

def resize_keep_ratio(frame, width):
    """Redimensiona mantendo a proporção."""
    h, w = frame.shape[:2]
    ratio = width / w
    return cv2.resize(frame, (width, int(h * ratio)))

def load_model():
    """Carrega o modelo YOLO."""
    print(f"📦 Carregando modelo: {MODEL_NAME}")
    try:
        model = YOLO(MODEL_NAME)
        print("✅ Modelo carregado com sucesso!")
        return model
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        raise

# =====================================================
# GERENCIADOR DE STREAM
# =====================================================

class StreamManager:
    def __init__(self):
        self.cap = None
        self.frame_queue = Queue(maxsize=2)
        self.running = False
        self.thread = None
        self.last_error = None
        
    def start(self):
        """Inicia o stream em uma thread separada."""
        self.running = True
        self.thread = threading.Thread(target=self._stream_worker)
        self.thread.daemon = True
        self.thread.start()
        
    def _connect_stream(self):
        """Tenta conectar ao stream usando múltiplas opções."""
        # Lista de URLs para tentar
        urls_to_try = []
        
        # 1. URLs das opções configuradas
        for option in STREAM_OPTIONS:
            if isinstance(option, str):
                urls_to_try.append(option)
            elif isinstance(option, int):
                # Webcam local
                try:
                    cap = cv2.VideoCapture(option)
                    if cap.isOpened():
                        print(f"✅ Webcam {option} encontrada!")
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        return cap
                except:
                    pass
        
        # 2. Verifica área de transferência
        clipboard_url = self._get_clipboard_url()
        if clipboard_url:
            urls_to_try.insert(0, clipboard_url)
        
        # 3. Tenta IP Webcam local
        ip_cam_url = self._get_ip_webcam_url()
        if ip_cam_url:
            urls_to_try.append(ip_cam_url)
        
        # 4. Adiciona URLs com IP local
        local_ip = self._get_ip_address()
        urls_to_try.extend([
            f"http://{local_ip}:8080/video",
            f"rtsp://{local_ip}:8554/stream",
            f"http://{local_ip}:8080/stream",
        ])
        
        # 5. Tenta VDO.ninja com diferentes formatos
        vdo_options = [
            STREAM_OPTIONS[0],
            STREAM_OPTIONS[0].replace("view=", "stream="),
            STREAM_OPTIONS[0] + "&stream=1",
        ]
        urls_to_try.extend(vdo_options)
        
        # Tentar cada URL
        for url in urls_to_try:
            print(f"🔍 Tentando: {url}")
            try:
                if "vdo.ninja" in str(url):
                    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                else:
                    cap = cv2.VideoCapture(url)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"✅ Conectado a: {url}")
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        return cap
                    else:
                        cap.release()
                        
                if "vdo.ninja" in str(url):
                    gst_cmd = f'gst-launch-1.0 souphttpsrc location="{url}" user-agent="{VDO_USER_AGENT}" ! decodebin ! videoconvert ! appsink'
                    try:
                        cap = cv2.VideoCapture(gst_cmd, cv2.CAP_GSTREAMER)
                        if cap.isOpened():
                            ret, frame = cap.read()
                            if ret and frame is not None:
                                print(f"✅ Conectado via GStreamer: {url}")
                                return cap
                    except:
                        pass
                        
            except Exception as e:
                print(f"❌ Erro ao tentar {url}: {e}")
                continue
        
        # Se nenhuma funcionou, tenta webcam local como último recurso
        print("🔄 Tentando webcam local...")
        for cam_id in [0, 1]:
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ Usando webcam local (ID: {cam_id})")
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    return cap
        
        raise RuntimeError("❌ Nenhuma fonte de vídeo disponível!")
    
    def _get_clipboard_url(self):
        """Tenta obter URL da área de transferência."""
        try:
            import pyperclip
            content = pyperclip.paste()
            if "vdo.ninja" in content:
                print(f"📋 URL encontrada na área de transferência: {content}")
                return content
        except:
            pass
        return None
    
    def _get_ip_webcam_url(self):
        """Tenta encontrar IP Webcam local."""
        for port in [8080, 8081, 8082, 8083]:
            url = f"http://localhost:{port}/video"
            try:
                response = requests.get(url, timeout=1)
                if response.status_code == 200:
                    return url
            except:
                continue
        return None
    
    def _get_ip_address(self):
        """Obtém o IP local."""
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "192.168.1.100"
    
    def _stream_worker(self):
        """Worker thread para capturar frames."""
        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    self.cap = self._connect_stream()
                    if self.cap is None:
                        time.sleep(1)
                        continue
                
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️ Stream perdido, reconectando...")
                    self.cap.release()
                    self.cap = None
                    time.sleep(1)
                    continue
                
                while self.frame_queue.qsize() >= 2:
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        pass
                
                self.frame_queue.put(frame)
                
            except Exception as e:
                self.last_error = str(e)
                print(f"❌ Erro no stream: {e}")
                time.sleep(1)
    
    def get_frame(self, timeout=1):
        """Obtém o último frame da fila."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except:
            return None
    
    def stop(self):
        """Para o stream."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
            self.cap = None

# =====================================================
# PROCESSADOR DE DETECÇÃO
# =====================================================

class FaceDetector:
    def __init__(self, model):
        self.model = model
        self.frame_count = 0
        self.faces_detected_total = 0
        self.faces_saved_total = 0
        self.last_save_time = {}
        self.last_inference_time = 0
        self.cached_results = None
        self.fps = 0
        self.fps_counter = 0
        self.fps_timer = time.time()
        
    def process_frame(self, frame):
        """Processa um frame e retorna o frame anotado e informações."""
        self.frame_count += 1
        now = time.time()
        faces_in_frame = 0
        
        # Limita FPS de inferência
        run_inference = False
        if now - self.last_inference_time >= 1 / MAX_INFERENCE_FPS:
            run_inference = True
            self.last_inference_time = now
        
        # Executa inferência
        if run_inference:
            self.cached_results = self.model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=CONFIDENCE,
                verbose=False
            )[0]
        
        results = self.cached_results
        
        # Processa detecções
        if results is not None and results.boxes is not None:
            for box in results.boxes:
                # Filtra apenas pessoas (classe 0)
                cls_id = int(box.cls[0])
                if cls_id != 0:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                face_w = x2 - x1
                face_h = y2 - y1
                
                if face_w < MIN_FACE_WIDTH or face_h < MIN_FACE_HEIGHT:
                    continue
                
                conf = float(box.conf[0])
                
                if hasattr(box, "id") and box.id is not None:
                    track_id = int(box.id.item())
                else:
                    track_id = -1
                
                face_crop = frame[
                    max(0, y1):min(frame.shape[0], y2),
                    max(0, x1):min(frame.shape[1], x2)
                ]
                
                if face_crop.size == 0:
                    continue
                
                faces_in_frame += 1
                self.faces_detected_total += 1
                
                # Desenha retângulo
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Texto com ID e confiança
                label = f"ID {track_id} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Salva a face
                if SAVE_FACES and not is_blurry(face_crop):
                    if track_id not in self.last_save_time:
                        self.last_save_time[track_id] = 0
                    
                    elapsed = now - self.last_save_time[track_id]
                    
                    if elapsed >= SAVE_COOLDOWN_SECONDS:
                        filename = save_face(face_crop, track_id, conf)
                        self.last_save_time[track_id] = now
                        self.faces_saved_total += 1
                        print(f"💾 Face salva: {filename}")
        
        # Atualiza FPS
        self.fps_counter += 1
        elapsed = time.time() - self.fps_timer
        if elapsed >= 1:
            self.fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_timer = time.time()
        
        # Adiciona overlay de informações
        self._add_overlay(frame, faces_in_frame)
        
        return frame, faces_in_frame
    
    def _add_overlay(self, frame, faces_in_frame):
        """Adiciona overlay de informações no frame."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
        
        infos = [
            f"FPS: {self.fps:.1f}",
            f"Faces no frame: {faces_in_frame}",
            f"Detecções totais: {self.faces_detected_total}",
            f"Faces salvas: {self.faces_saved_total}",
            f"Frame: {self.frame_count}",
            f"Modelo: {MODEL_NAME}"
        ]
        
        for i, txt in enumerate(infos):
            cv2.putText(frame, txt, (10, 20 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def reset_counters(self):
        """Reseta os contadores."""
        self.faces_detected_total = 0
        self.faces_saved_total = 0
        self.last_save_time = {}
        print("🔄 Contadores resetados!")
    
    def get_stats(self):
        """Retorna estatísticas do processamento."""
        return {
            'frames': self.frame_count,
            'detections': self.faces_detected_total,
            'saved': self.faces_saved_total,
            'fps': self.fps
        }

# =====================================================
# FUNÇÃO PRINCIPAL
# =====================================================

def main():
    print("="*60)
    print("   DETECÇÃO FACIAL COM YOLO + STREAM")
    print("="*60)
    
    # Carrega modelo
    print("\n📦 Carregando modelo YOLO...")
    model = load_model()
    
    # Inicia stream
    print("\n📡 Conectando ao stream...")
    stream = StreamManager()
    stream.start()
    time.sleep(2)
    
    # Verifica conexão
    test_frame = stream.get_frame(timeout=3)
    if test_frame is None:
        print("⚠️ Nenhum frame recebido. Verifique a fonte.")
        print("   Dicas:")
        print("   - Certifique-se de que o celular está compartilhando a câmera")
        print("   - Atualize o link do VDO.ninja")
        print("   - Tente usar IP Webcam ou webcam local")
    else:
        print("✅ Stream conectado!")
    
    # Cria detector e janela
    detector = FaceDetector(model)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    
    print("\n" + "="*60)
    print("🎯 DETECÇÃO EM EXECUÇÃO")
    print("   Pressione 'q' para sair")
    print("   Pressione 's' para salvar frame")
    print("   Pressione 'r' para resetar contadores")
    print("="*60 + "\n")
    
    try:
        while True:
            # Obtém frame
            frame = stream.get_frame(timeout=1)
            
            if frame is None:
                print("⏳ Aguardando frame...")
                continue
            
            # Processa frame
            processed_frame, _ = detector.process_frame(frame)
            
            # Exibe
            display = resize_keep_ratio(processed_frame, DISPLAY_WIDTH)
            cv2.imshow(WINDOW_NAME, display)
            
            # Controles
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n🛑 Saindo...")
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"frame_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Frame salvo: {filename}")
            elif key == ord('r'):
                detector.reset_counters()
    
    except KeyboardInterrupt:
        print("\n🛑 Interrompido pelo usuário.")
    
    finally:
        # Finalização
        stream.stop()
        cv2.destroyAllWindows()
        
        stats = detector.get_stats()
        print("\n" + "="*60)
        print("📊 RESUMO FINAL")
        print("="*60)
        print(f"Frames processados: {stats['frames']}")
        print(f"Total de detecções: {stats['detections']}")
        print(f"Faces salvas: {stats['saved']}")
        
        if stats['detections'] > 0 and stats['frames'] > 0:
            economia = 100 * (1 - min(1, stats['frames'] / max(1, stats['detections'])))
            print(f"Economia de processamento: {economia:.1f}%")
        
        if stats['saved'] > 0:
            print(f"\n📁 Faces salvas em: {os.path.abspath(SAVE_DIR)}")
        
        print("="*60)
        print("✅ Processamento concluído!")

if __name__ == "__main__":
    main()