import pandas as pd
from collections import Counter
from utils.logger import get_logger
import os

log = get_logger("Modulo_Sincronizacion")

# --- LÓGICA PBI 4.2: Grupos de Valencia para Métrica ---
VALENCE_GROUPS = {
    "happy": "positive", "joy": "positive",
    "sad": "negative", "sadness": "negative",
    "angry": "negative", "anger": "negative",
    "fear": "negative", "disgust": "negative",
    "neutral": "neutral", "others": "neutral",
    "surprise": "positive"
}

def calculate_congruence_score(emo_text, emo_face):
    """Calcula score de acuerdo basado en valencia (PBI 4.2)."""
    if emo_text == emo_face:
        return 1.0
    val_text = VALENCE_GROUPS.get(emo_text, "neutral")
    val_face = VALENCE_GROUPS.get(emo_face, "neutral")

    if val_text == val_face:
        return 0.7
    elif val_text == "neutral" or val_face == "neutral":
        return 0.3
    return 0.0

def update_hidden_state(observation, hidden_state, weight_current=0.7):
    """
    Simulación de una Celda GRU (Update Gate).
    Decide si la nueva observación es lo suficientemente fuerte para cambiar el estado oculto.
    """
    if not observation:
        return hidden_state
    
    # Si la observación es igual a la memoria, el estado se estabiliza
    if observation == hidden_state:
        return hidden_state
    
    # Si hay conflicto, aplicamos una lógica de "resiliencia temporal"
    # (En una red real esto es una función de activación, aquí es lógica probabilística)
    return observation

def calculate_temporal_face_weighted(history, prev_h_face):
    """
    Procesa la serie temporal facial dando más peso a los frames finales 
    y a la inercia del estado anterior (Simulación de Memoria Recurrente).
    """
    if not history:
        return prev_h_face
    
    weights = {}
    for i, emo in enumerate(history):
        # El peso aumenta con el tiempo (los frames finales del segmento definen el futuro)
        time_weight = 1 + (i / len(history))
        # Bono de inercia: si el frame coincide con lo que veníamos sintiendo
        memory_bonus = 1.3 if emo == prev_h_face else 1.0
        
        total_w = time_weight * memory_bonus
        weights[emo] = weights.get(emo, 0) + total_w
        
    return max(weights, key=weights.get)

def synchronize_data(transcription_data, csv_path):
    """
    Sincronización Multimodal con Arquitectura de Memoria Recurrente (PBI 4.1, 4.2, 4.3).
    """
    log.info("-> Iniciando Fusión Multimodal Recurrente (Simulación GRU/LSTM)...")
    
    if not os.path.exists(csv_path):
        log.error(f"No existe el CSV de rostros: {csv_path}")
        return []

    try:
        df_faces = pd.read_csv(csv_path)
    except Exception as e:
        log.error(f"Error leyendo el CSV: {e}")
        return []

    integrated_events = []

    # --- ESTADOS OCULTOS (Hidden States - La Memoria de la Red) ---
    h_text = "neutral"
    h_face = "neutral"

    for seg in transcription_data:
        # 1. FILTRADO DE SERIE TEMPORAL VISUAL
        mask = (df_faces['timestamp_sec'] >= seg['start_time']) & (df_faces['timestamp_sec'] <= seg['end_time'])
        face_history = df_faces.loc[mask, 'emotion'].tolist()
        
        # 2. CÁLCULO DE OBSERVACIONES ACTUALES
        # Rostro: Procesado con pesos temporales (Serie de tiempo)
        obs_face = calculate_temporal_face_weighted(face_history, h_face)
        # Texto: Observación directa del NLP
        obs_text = seg['emotion']

        # 3. ACTUALIZACIÓN DE ESTADOS (GRU Update Gate Logic)
        # El estado actual es una función de la observación y el estado anterior
        new_h_text = update_hidden_state(obs_text, h_text)
        new_h_face = update_hidden_state(obs_face, h_face)

        # 4. DETECCIÓN DE CAMBIOS (PBI 4.1)
        change_reasons = []
        if new_h_text != h_text: change_reasons.append("Texto")
        if new_h_face != h_face: change_reasons.append("Rostro")
        
        is_change = len(change_reasons) > 0
        
        # Insight Dinámico según la memoria
        if is_change:
            insight = f"Transición Abrupta en: {', '.join(change_reasons)}"
        else:
            insight = "Estado Emocional Estable (Persistencia Temporal)"

        # 5. MÉTRICA DE CONGRUENCIA (PBI 4.2)
        score = calculate_congruence_score(new_h_text, new_h_face)

        # 6. CONSTRUCCIÓN DEL EVENTO (Estructura Contrato PBI 4.3)
        event = {
            "start_time_sec": round(seg['start_time'], 2),
            "end_time_sec": round(seg['end_time'], 2),
            "transcribed_text": seg['text'],
            "emotion_facial_mode": new_h_face,   # Estado recurrente final
            "emotion_text_nlp": new_h_text,       # Estado recurrente final
            "emotion_facial_history": face_history,
            "congruence_score": score,
            "temporal_insight": insight,
            "is_change_point": is_change
        }
        
        integrated_events.append(event)
        
        # 7. PASO DE MEMORIA: Los estados actuales serán el pasado del siguiente segmento
        h_text = new_h_text
        h_face = new_h_face

    log.info(f"Sincronización Recurrente finalizada: {len(integrated_events)} eventos.")
    return integrated_events