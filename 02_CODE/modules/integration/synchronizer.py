import pandas as pd
from collections import Counter
from utils.logger import get_logger

log = get_logger("Modulo_Sincronizacion")

def synchronize_data(transcription_data, csv_path):
    log.info("-> Ejecutando detección de cambios Multimodal (PBI 4.1 Avanzado)...")
    df_faces = pd.read_csv(csv_path)
    integrated = []
    
    prev_text = None
    prev_face = None

    for seg in transcription_data:
        mask = (df_faces['timestamp_sec'] >= seg['start_time']) & (df_faces['timestamp_sec'] <= seg['end_time'])
        segment_faces = df_faces.loc[mask]
        
        face_mode = Counter(segment_faces['emotion'].tolist()).most_common(1)[0][0] if not segment_faces.empty else "neutral"
        
        # --- NUEVA LÓGICA DE DETECCIÓN AVANZADA ---
        change_reasons = []
        
        # 1. ¿Cambió el texto?
        if prev_text and seg['emotion'] != prev_text:
            change_reasons.append("Texto")
            
        # 2. ¿Cambió la cara? (Esto detectará el salto a 'angry' en tu video)
        if prev_face and face_mode != prev_face:
            change_reasons.append("Rostro")
            
        # Determinar si es un punto de cambio
        is_change = len(change_reasons) > 0
        insight = f"Cambio en: {', '.join(change_reasons)}" if is_change else "Estable"

        event = {
            "start_time_sec": seg['start_time'],
            "end_time_sec": seg['end_time'],
            "text": seg['text'],
            "emotion_text_nlp": seg['emotion'],
            "emotion_facial_mode": face_mode,
            "congruence": 1.0 if seg['emotion'] == face_mode else 0.0,
            "temporal_insight": insight,
            "is_change_point": is_change # Ahora marca si cambió CUALQUIERA de los dos
        }
        
        integrated.append(event)
        prev_text = seg['emotion']
        prev_face = face_mode

    return integrated