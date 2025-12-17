import cv2
import pandas as pd
import os
import sys

# --- AJUSTE DE IMPORTACIONES ---
# Agregamos el directorio raíz (02_CODE) al path para poder importar utils
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", ".."))

from deepface import DeepFace
from utils.logger import get_logger
from utils.helpers import validate_input_file, create_output_directory

# --- CONFIGURACIÓN ---
# Inicializamos el logger específico para este módulo (PBI 2.6)
log = get_logger("CNN_Extractor")

# Rutas Base
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Apunta a 02_CODE
ROOT_DIR = os.path.dirname(BASE_DIR) # Apunta a SISINTFINAL

# Video por defecto para pruebas (puedes cambiarlo aquí o pasar otro argumento)
DEFAULT_VIDEO_SOURCE = os.path.join(ROOT_DIR, "01_DATA", "raw", "video_03.mp4")

def extract_faces_from_video(video_path, sample_rate=15):
    """
    PBI 2.1: Procesa el video frame a frame y genera una serie temporal de emociones en CSV.
    
    Args:
        video_path (str): Ruta al archivo de video.
        sample_rate (int): Procesa 1 de cada N frames (15 es aprox cada 0.5s en videos de 30fps).
    """
    
    # 1. Validación de Entrada
    if not validate_input_file(video_path):
        log.error(f"No se puede iniciar la extracción. Video no válido: {video_path}")
        return

    # --- GENERACIÓN DE NOMBRE DINÁMICO DE SALIDA ---
    # Obtener el nombre del archivo sin extensión (ej: "video_entrevista_3")
    filename = os.path.basename(video_path)
    filename_without_ext = os.path.splitext(filename)[0]
    
    # Crear nombre del CSV (ej: "video_entrevista_3_faces.csv")
    csv_filename = f"{filename_without_ext}_faces.csv"
    
    # Definir la ruta completa de salida en una subcarpeta organizada
    output_csv_path = os.path.join(ROOT_DIR, "05_OUTPUTS", "series_temporales", csv_filename)
    # -----------------------------------------------

    log.info(f"Iniciando extracción DeepFace (Sample Rate: {sample_rate})")
    log.info(f"Video de entrada: {filename}")
    log.info(f"Archivo de salida será: {output_csv_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Error crítico: OpenCV no pudo abrir el archivo de video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info(f"Metadatos del Video -> FPS: {fps:.2f}, Total Frames: {total_frames_video}")
    
    frame_data = []
    frame_count = 0
    processed_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Procesamos solo cada 'sample_rate' frames para optimizar tiempo
        if frame_count % sample_rate == 0:
            try:
                # DeepFace analysis
                # backend='opencv' es más rápido para CPU que retinaface
                result = DeepFace.analyze(
                    img_path=frame, 
                    actions=['emotion'], 
                    enforce_detection=False, # Importante: no fallar si no hay cara en un frame
                    detector_backend='opencv', 
                    silent=True
                )
                
                if result:
                    # DeepFace puede devolver múltiples caras, tomamos la primera
                    face_data = result[0]
                    emotion = face_data['dominant_emotion']
                    confidence = face_data['emotion'][emotion]
                    
                    # Calcular tiempo exacto en segundos
                    timestamp = frame_count / fps
                    
                    data = {
                        "frame": frame_count,
                        "timestamp_sec": round(timestamp, 2),
                        "emotion": emotion,
                        "confidence": round(confidence, 2)
                    }
                    frame_data.append(data)
                    processed_count += 1
                    
                    # Log de progreso en consola
                    if processed_count % 20 == 0:
                        print(f"   -> Procesando... {processed_count} muestras capturadas (T: {timestamp:.2f}s)")

            except Exception as e:
                # Si falla un frame específico, lo ignoramos y seguimos
                # log.debug(f"Frame {frame_count} ignorado: {e}") 
                pass

        frame_count += 1

    cap.release()
    
    # 2. Guardado de Salida
    if frame_data:
        try:
            # Asegurar que el directorio de salida exista
            create_output_directory(os.path.dirname(output_csv_path))
            
            df = pd.DataFrame(frame_data)
            df.to_csv(output_csv_path, index=False)
            
            log.info(f"¡ÉXITO! Procesamiento finalizado.")
            log.info(f"Frames procesados: {processed_count}/{frame_count}")
            log.info(f"Serie temporal guardada en: {output_csv_path}")
        except Exception as e:
            log.error(f"Error al guardar el CSV: {e}")
    else:
        log.warning("El proceso finalizó pero NO se detectaron rostros. Revisa la iluminación o el backend.")

if __name__ == "__main__":
    # Ejecución directa con el video por defecto
    extract_faces_from_video(DEFAULT_VIDEO_SOURCE)