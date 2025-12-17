import os
import cv2
import datetime
from utils.logger import get_logger

# Inicializamos el logger para este módulo
log = get_logger("Utils_Helpers")

def validate_input_file(filepath):
    """
    Verifica si un archivo existe y es legible.
    Retorna True si existe, False si no, y loguea el resultado.
    """
    if not filepath:
        log.error("La ruta del archivo está vacía.")
        return False
    
    if os.path.exists(filepath):
        log.info(f"Archivo validado correctamente: {filepath}")
        return True
    else:
        log.error(f"No se encontró el archivo: {filepath}")
        return False

def get_video_properties(video_path):
    """
    Usa OpenCV para extraer FPS, cantidad de frames y duración del video.
    Útil para la sincronización temporal (PBI de Integración).
    """
    if not validate_input_file(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error(f"No se pudo abrir el video con OpenCV: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calcular duración en segundos
    duration_sec = 0
    if fps > 0:
        duration_sec = total_frames / fps

    cap.release()

    properties = {
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 2)
    }
    
    log.info(f"Propiedades del video extraídas: {properties}")
    return properties

def format_timestamp(seconds):
    """
    Convierte segundos (float) a formato string HH:MM:SS.
    Ejemplo: 75.5 -> "00:01:15"
    """
    try:
        return str(datetime.timedelta(seconds=int(seconds)))
    except Exception as e:
        log.warning(f"Error formateando tiempo {seconds}: {e}")
        return "00:00:00"

def create_output_directory(directory_path):
    """
    Crea una carpeta si no existe. Útil para la carpeta 04_OUTPUTS.
    """
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            log.info(f"Directorio creado: {directory_path}")
        except OSError as e:
            log.error(f"Error creando directorio {directory_path}: {e}")