import pandas as pd
from collections import Counter
from utils.logger import get_logger
import os

log = get_logger("Modulo_Sincronizacion")

# --- LÓGICA PBI 4.2: Grupos de Valencia para Métrica Avanzada ---
VALENCE_GROUPS = {
    "happy": "positive",
    "joy": "positive",
    "sad": "negative",
    "sadness": "negative",
    "angry": "negative",
    "anger": "negative",
    "fear": "negative",
    "disgust": "negative",
    "neutral": "neutral",
    "others": "neutral",
    "surprise": "positive"
}

def calculate_congruence_score(emo_text, emo_face):
    """
    Calcula un score de 0.0 a 1.0 basado en la valencia (PBI 4.2).
    """
    if emo_text == emo_face:
        return 1.0  # Coincidencia exacta
    
    val_text = VALENCE_GROUPS.get(emo_text, "neutral")
    val_face = VALENCE_GROUPS.get(emo_face, "neutral")

    if val_text == val_face:
        return 0.7  # Misma valencia (ej. triste y enojado)
    elif val_text == "neutral" or val_face == "neutral":
        return 0.3  # Uno es neutral y el otro tiene carga emocional
    else:
        return 0.0  # Conflicto total (ej. Feliz vs Enojado)

def synchronize_data(transcription_data, csv_path):
    """
    Sincroniza audio y video, detecta cambios (4.1) y calcula score (4.2).
    Sigue la estructura de contrato PBI 4.3.
    """
    log.info("-> Sincronizando datos con estructura avanzada (History + Insights)...")
    
    if not os.path.exists(csv_path):
        log.error(f"No existe el CSV de rostros: {csv_path}")
        return []

    try:
        df_faces = pd.read_csv(csv_path)
    except Exception as e:
        log.error(f"Error leyendo el CSV: {e}")
        return []

    integrated_events = []
    prev_text = None
    prev_face = None

    for seg in transcription_data:
        # 1. Filtrar frames del CSV por el intervalo de tiempo del audio
        mask = (df_faces['timestamp_sec'] >= seg['start_time']) & (df_faces['timestamp_sec'] <= seg['end_time'])
        segment_faces = df_faces.loc[mask]
        
        # --- NUEVO: CAPTURAR HISTORIAL DE EMOCIONES (PBI 4.3) ---
        face_history = segment_faces['emotion'].tolist() if not segment_faces.empty else []
        
        # Obtener emoción facial dominante (Moda)
        if face_history:
            face_mode = Counter(face_history).most_common(1)[0][0]
            face_conf = segment_faces['confidence'].mean()
        else:
            face_mode = "neutral"
            face_conf = 0.0

        # 2. LÓGICA PBI 4.1 (Detección de Cambios Multimodal)
        change_reasons = []
        if prev_text and seg['emotion'] != prev_text:
            change_reasons.append("Texto")
        if prev_face and face_mode != prev_face:
            change_reasons.append("Rostro")
        
        is_change = len(change_reasons) > 0
        insight = f"Cambio detectado en: {', '.join(change_reasons)}" if is_change else "Estable"

        # 3. LÓGICA PBI 4.2 (Métrica de Congruencia Avanzada)
        score = calculate_congruence_score(seg['emotion'], face_mode)

        # 4. CONSTRUCCIÓN DEL EVENTO (Según Estructura de Contrato Solicitada)
        event = {
            "start_time_sec": round(seg['start_time'], 2),
            "end_time_sec": round(seg['end_time'], 2),
            "transcribed_text": seg['text'],
            "emotion_facial_mode": face_mode,
            "emotion_facial_history": face_history, # Evidencia frame a frame
            "emotion_text_nlp": seg['emotion'],
            "congruence_score": score,
            "temporal_insight": insight,
            "is_change_point": is_change # Mantiene compatibilidad con visualizer
        }
        
        integrated_events.append(event)
        
        # Actualizar estados previos para la siguiente iteración
        prev_text = seg['emotion']
        prev_face = face_mode

    log.info(f"Sincronización finalizada: {len(integrated_events)} eventos generados.")
    return integrated_events