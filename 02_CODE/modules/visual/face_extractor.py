import cv2
import pandas as pd
import os
from deepface import DeepFace
from utils.logger import get_logger

# Configuración
log = get_logger("CNN_Extractor")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "04_OUTPUTS", "cnn_time_series.csv")

# Define aquí qué video vas a procesar (debe coincidir con el de audio)
VIDEO_SOURCE = os.path.join(BASE_DIR, "01_DATA", "raw", "video_03.mp4")

def extract_faces_from_video(video_path, sample_rate=15):
    """
    PBI 2.1: Procesa el video y genera una serie temporal de emociones en CSV.
    sample_rate: Procesa 1 de cada N frames (para optimizar tiempo CPU).
    """
    if not os.path.exists(video_path):
        log.error(f"Video no encontrado: {video_path}")
        return

    log.info(f"Iniciando extracción DeepFace en: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_data = []
    frame_count = 0
    processed_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Procesamos solo cada 'sample_rate' frames
        if frame_count % sample_rate == 0:
            try:
                # DeepFace analysis
                # enforce_detection=False para que no se detenga si tapa la cara un segundo
                result = DeepFace.analyze(
                    img_path=frame, 
                    actions=['emotion'], 
                    enforce_detection=False,
                    detector_backend='opencv', # Backend rápido
                    silent=True
                )
                
                # DeepFace devuelve una lista, tomamos el primer rostro
                if result:
                    emotion = result[0]['dominant_emotion']
                    confidence = result[0]['emotion'][emotion]
                    
                    # Calcular tiempo exacto
                    timestamp = frame_count / fps
                    
                    data = {
                        "frame": frame_count,
                        "timestamp_sec": round(timestamp, 2),
                        "emotion": emotion,
                        "confidence": confidence
                    }
                    frame_data.append(data)
                    processed_count += 1
                    
                    if processed_count % 10 == 0:
                        print(f"-> Procesados {processed_count} frames... (Tiempo: {timestamp:.2f}s)")

            except Exception as e:
                # Si falla un frame puntual, lo ignoramos y seguimos
                pass

        frame_count += 1

    cap.release()
    
    # Guardar a CSV
    if frame_data:
        df = pd.DataFrame(frame_data)
        df.to_csv(OUTPUT_CSV_PATH, index=False)
        log.info(f"Serie temporal guardada exitosamente: {OUTPUT_CSV_PATH} ({len(df)} registros)")
    else:
        log.warning("No se extrajeron datos faciales. Revisa el video o la iluminación.")

if __name__ == "__main__":
    extract_faces_from_video(VIDEO_SOURCE)