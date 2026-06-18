# Acesse VDO.ninja pelo celular e selecione "Share your Camera"
# Para mudar qual das câmeras, mudar "Video Source"
# Ir em "START" e substituir pelo link em verde que esta no site

# pip install ultralytics opencv-python numpy

import cv2
import numpy as np
import time
from ultralytics import YOLO

# ----------------------------- CONFIGURAÇÕES -----------------------------
# Link do VDO.ninja (substitua pelo seu link)
VDO_LINK = "https://vdo.ninja/?view=vpHxLKi"

# Parâmetros do detector de movimento (Lucas-Kanade)
MIN_FEATURES = 100               # mínimo de pontos para rastrear
QUALITY_LEVEL = 0.3              # qualidade dos cantos (0..1)
MIN_DISTANCE = 7                 # distância mínima entre cantos
LK_WIN_SIZE = (15, 15)           # tamanho da janela de busca
MAX_LEVEL = 2                    # níveis da pirâmide
MOTION_THRESHOLD = 2.0           # magnitude média mínima (pixels) para considerar movimento
REINIT_FRAMES = 30               # a cada N frames, re‑detecta os pontos de interesse

# Modelo YOLO
MODEL_NAME = "yolov8n.pt"        # nano (mais leve); troque para yolov8s.pt etc. se preferir
CONFIDENCE = 0.5                 # confiança mínima para mostrar detecções

# Configuração de exibição
WINDOW_NAME = "Reconhecimento Facial - Stream Celular"
RESIZE_WIDTH = 800  # Largura para redimensionar a janela (mantém proporção)

# --------------------------- FUNÇÕES AUXILIARES ---------------------------
def draw_detections(frame, results):
    """Desenha as bounding boxes e labels das detecções YOLO no frame."""
    if results.boxes is None:
        return
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        
        # Nome da classe detectada
        class_name = results.names[cls_id]
        
        # Se for pessoa (classe 0) ou qualquer outra, destacamos diferente
        if class_name == "person":
            color = (0, 255, 0)  # Verde para pessoas
        else:
            color = (255, 255, 0)  # Amarelo para outros objetos
            
        label = f"{class_name} {conf:.2f}"
        
        # Desenha retângulo
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Desenha fundo do texto para melhor legibilidade
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        
        # Desenha texto
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

def draw_tracking_points(frame, points, color=(0, 255, 255)):
    """Desenha os pontos rastreados no frame."""
    if points is not None and len(points) > 0:
        for pt in points:
            x, y = pt.ravel()
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)

def resize_frame(frame, width):
    """Redimensiona o frame mantendo a proporção."""
    height = int(frame.shape[0] * (width / frame.shape[1]))
    return cv2.resize(frame, (width, height))

def create_overlay(frame):
    """Cria uma overlay com informações do sistema."""
    overlay = frame.copy()
    # Adiciona uma barra semi-transparente no topo
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
    return cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

# ----------------------------- INICIALIZAÇÃO -----------------------------
# Carrega o modelo YOLO
print("Carregando YOLO...")
model = YOLO(MODEL_NAME)

# Abre o stream do VDO.ninja
print(f"Conectando ao stream: {VDO_LINK}")
print("Certifique-se de que o celular está compartilhando a câmera via VDO.ninja")

cap = cv2.VideoCapture(VDO_LINK)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduz buffer para menor latência

if not cap.isOpened():
    raise IOError(f"Não foi possível abrir o stream: {VDO_LINK}")

# Variáveis do rastreador Lucas‑Kanade
prev_gray = None
prev_pts = None
frame_count = 0
inference_count = 0
fps_counter = 0
fps_start_time = time.time()
current_fps = 0

# Parâmetros para detecção de cantos (Shi‑Tomasi)
feature_params = dict(
    maxCorners=200,
    qualityLevel=QUALITY_LEVEL,
    minDistance=MIN_DISTANCE,
    blockSize=7
)

# Parâmetros do Lucas‑Kanade
lk_params = dict(
    winSize=LK_WIN_SIZE,
    maxLevel=MAX_LEVEL,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Cria a janela de exibição
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, RESIZE_WIDTH, int(RESIZE_WIDTH * 0.75))

print("\n=== Stream iniciado! ===")
print("Pressione 'q' para sair")
print("Pressione 's' para salvar o frame atual")
print("Pressione 'r' para reiniciar a detecção de movimento")
print("========================\n")

# -------------------------- LOOP PRINCIPAL --------------------------
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Stream finalizado ou desconectado. Tentando reconectar...")
            time.sleep(2)
            cap = cv2.VideoCapture(VDO_LINK)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_detected = False

        # Inicializa ou reinicializa os pontos de interesse
        if prev_gray is None or frame_count % REINIT_FRAMES == 0:
            # Detecta cantos para rastrear
            prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
            if prev_pts is not None:
                prev_pts = prev_pts.reshape(-1, 1, 2)
            prev_gray = gray.copy()
            # No primeiro frame, forçamos uma inferência
            if frame_count == 1:
                motion_detected = True
        else:
            # Calcula o fluxo óptico Lucas‑Kanade
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, prev_pts, None, **lk_params
            )
            # Filtra apenas os pontos rastreados com sucesso
            if curr_pts is not None and status is not None:
                good_new = curr_pts[status.flatten() == 1]
                good_old = prev_pts[status.flatten() == 1]

                if len(good_new) >= MIN_FEATURES:
                    # Magnitude do deslocamento dos pontos
                    motion_vectors = good_new - good_old
                    magnitudes = np.sqrt((motion_vectors ** 2).sum(axis=1))
                    mean_magnitude = np.mean(magnitudes)

                    # Decisão do gatilho
                    if mean_magnitude > MOTION_THRESHOLD:
                        motion_detected = True

                    # Atualiza os pontos para o próximo frame
                    prev_pts = good_new.reshape(-1, 1, 2)
                else:
                    # Poucos pontos rastreados – reinicializa
                    prev_gray = None
            else:
                prev_gray = None

            prev_gray = gray.copy()

        # ---------- INFERÊNCIA YOLO SE O GATILHO FOI ATIVADO ----------
        if motion_detected:
            # Usa track para manter IDs consistentes entre frames
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, conf=CONFIDENCE)[0]
            inference_count += 1
            
            # Desenha as detecções
            draw_detections(frame, results)
            
            # Desenha os pontos de movimento
            draw_tracking_points(frame, prev_pts, (0, 255, 255))
            
            # Adiciona indicador de MOVIMENTO DETECTADO
            status_text = "MOVIMENTO DETECTADO"
            status_color = (0, 0, 255)  # Vermelho
        else:
            # Sem movimento detectado
            draw_tracking_points(frame, prev_pts, (255, 0, 0))
            status_text = "AGUARDANDO MOVIMENTO..."
            status_color = (255, 255, 0)  # Amarelo

        # Calcula FPS
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            current_fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()

        # Cria overlay com informações
        frame_with_overlay = create_overlay(frame)
        
        # Adiciona informações na barra superior
        info_lines = [
            f"Status: {status_text}",
            f"Frame: {frame_count} | Inferencias: {inference_count}",
            f"FPS: {current_fps} | Pontos: {len(prev_pts) if prev_pts is not None else 0}"
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = 20 + (i * 20)
            cv2.putText(frame_with_overlay, line, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Redimensiona para exibição
        display_frame = resize_frame(frame_with_overlay, RESIZE_WIDTH)

        # Exibe o frame
        cv2.imshow(WINDOW_NAME, display_frame)

        # ---------- CONTROLES DO TECLADO ----------
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Saindo...")
            break
        elif key == ord('s'):
            # Salva o frame atual
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"captura_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Frame salvo como: {filename}")
        elif key == ord('r'):
            # Reinicia a detecção de movimento
            prev_gray = None
            prev_pts = None
            print("Detecção de movimento reiniciada!")

except KeyboardInterrupt:
    print("\nStream interrompido pelo usuário.")
    
finally:
    # --------------------------- FINALIZAÇÃO ---------------------------
    cap.release()
    cv2.destroyAllWindows()
    
    # Relatório de otimização
    print("\n" + "="*50)
    print("PROCESSAMENTO CONCLUÍDO")
    print("="*50)
    print(f"Total de frames processados: {frame_count}")
    print(f"Frames com inferência YOLO: {inference_count}")
    if frame_count > 0:
        economia = 100 * (1 - inference_count / frame_count)
        print(f"Economia de processamento: {economia:.1f}%")
    print("="*50)