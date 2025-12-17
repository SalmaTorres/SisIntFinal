import pandas as pd
import sys
import os
from collections import Counter # <--- CAMBIO: Usamos Counter en lugar de statistics

# --- IMPORTACIONES UTILS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", "..", ".."))

from utils.logger import get_logger

log = get_logger("Modulo_Sincronizacion")

def calculate_congruence(emotion_text, emotion_face):
    """
    Calcula si hay congruencia (1.0) o no (0.0) entre texto y cara.
    """
    # Mapeo básico de sinónimos para asegurar compatibilidad
    map_emotions = {
        "joy": "happy",
        "happiness": "happy",
        "sadness": "sad",
        "anger": "angry",
        "fear": "fear",
        "surprise": "surprise",
        "neutral": "neutral"
    }
    
    # Normalizar strings
    et = str(emotion_text).lower()
    ef = str(emotion_face).lower()
    
    # Traducir usando el mapa
    et_mapped = map_emotions.get(et, et)
    ef_mapped = map_emotions.get(ef, ef)
    
    return 1.0 if et_mapped == ef_mapped else 0.0

def synchronize_multimodal_data(transcription_data: list, faces_csv_path: str) -> list:
    """
    IMPLEMENTACIÓN REAL PBI 3.1 (Corregida con Counter)
    """
    log.info(f"--- INICIANDO SINCRONIZACIÓN REAL ---")
    
    # 1. Cargar el CSV de Caras
    if not os.path.exists(faces_csv_path):
        log.error(f"No se encontró el CSV: {faces_csv_path}")
        return []
    
    try:
        df_faces = pd.read_csv(faces_csv_path)
        log.info(f"CSV cargado con {len(df_faces)} frames.")
    except Exception as e:
        log.error(f"Error leyendo CSV: {e}")
        return []

    integrated_events = []

    # 2. Recorrer cada segmento de audio
    for segment in transcription_data:
        start_t = segment['start_time']
        end_t = segment['end_time']
        
        # Obtener emoción de texto de manera segura
        text_emotion = segment.get('emotion', 'neutral') 
        # Si por alguna razón viene vacío, forzar neutral
        if not text_emotion: text_emotion = 'neutral'

        # 3. Filtrar el DataFrame por tiempo
        mask = (df_faces['timestamp_sec'] >= start_t) & (df_faces['timestamp_sec'] <= end_t)
        segment_faces = df_faces.loc[mask]
        
        # Valores por defecto
        face_emotion_mode = "neutral"
        face_confidence_avg = 0.0
        face_history = []
        
        if not segment_faces.empty:
            # Extraer lista de emociones
            emotions_list = segment_faces['emotion'].tolist()
            face_history = emotions_list
            
            # --- CORRECCIÓN CLAVE: Usar Counter para la Moda ---
            if emotions_list:
                # most_common(1) devuelve una lista [(elemento, cuenta)]
                # Ejemplo: [('happy', 5)]
                # Tomamos el primer elemento ([0][0])
                face_emotion_mode = Counter(emotions_list).most_common(1)[0][0]
            
            # Calcular confianza promedio
            face_confidence_avg = segment_faces['confidence'].mean()
        else:
            # Si no hay caras detectadas en ese lapso
            face_emotion_mode = "no_face_detected"

        # 4. Calcular Congruencia
        congruence = calculate_congruence(text_emotion, face_emotion_mode)

        # --- AGREGA ESTO PARA VERIFICAR EN VIVO ---
        if not segment_faces.empty:
            log.info(f" > Seg [{start_t:.1f}s - {end_t:.1f}s]: Texto='{text_emotion}' vs Cara='{face_emotion_mode}' ({len(segment_faces)} frames). Congruencia: {congruence}")
        else:
            log.warning(f" > Seg [{start_t:.1f}s - {end_t:.1f}s]: SIN CARAS DETECTADAS.")
        # ------------------------------------------

        # 5. Construir Evento
        event = {
            "start_time_sec": start_t,
            "end_time_sec": end_t,
            "transcribed_text": segment['text'],
            
            "emotion_facial_mode": face_emotion_mode,
            "emotion_facial_history": face_history,
            "confidence_facial_mode": round(face_confidence_avg, 2),
            
            "emotion_text_nlp": text_emotion,
            # Aseguramos que confidence_nlp exista
            "confidence_nlp": segment.get('confidence', 0.0),
            
            "congruence_score": congruence,
            "temporal_insight": "" 
        }
        
        integrated_events.append(event)

    log.info(f"Sincronización finalizada. {len(integrated_events)} eventos unificados.")
    return integrated_events