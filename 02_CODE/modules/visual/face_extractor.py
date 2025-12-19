import os
import cv2
import pandas as pd
from deepface import DeepFace
from utils.logger import get_logger
from utils.helpers import validate_input_file, create_output_directory

log = get_logger("Modulo_DeepFace")

def extract_faces_from_video(video_path, csv_path, sample_rate=30):
    if os.path.exists(csv_path):
        log.info(f"Serie temporal encontrada: {os.path.basename(csv_path)}. Saltando.")
        return True

    if not validate_input_file(video_path): return False
    create_output_directory(os.path.dirname(csv_path))

    log.info("Iniciando análisis facial frame a frame...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = 0
    data = []  
    
    while True:
        # En lugar de leer cada frame, saltamos directamente al que nos interesa
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count) 
        ret, frame = cap.read()
        if not ret: break

        try:
            small_frame = cv2.resize(frame, (640, 480))
            result = DeepFace.analyze(small_frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv', silent=True)
            if result:
                res = result[0]
                data.append({
                    "timestamp_sec": round(frame_count / fps, 2),
                    "emotion": res['dominant_emotion'],
                    "confidence": res['emotion'][res['dominant_emotion']]
            })
        except Exception as e:
            log.warning(f"Frame {frame_count} no procesable: {e}")

        frame_count += sample_rate
    
    cap.release()
    pd.DataFrame(data).to_csv(csv_path, index=False)
    log.info(f"Análisis facial completado. CSV en: {csv_path}")
    return True