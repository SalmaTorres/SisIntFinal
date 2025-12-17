import pandas as pd
import sys
import os

# --- IMPORTACIONES UTILS ---
# Truco para importar desde directorios superiores
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", "..", ".."))

from utils.logger import get_logger

log = get_logger("Modulo_Sincronizacion")

def synchronize_multimodal_data(transcription_data: list, faces_csv_path: str) -> list:
    """
    [PBI 3.1 - A IMPLEMENTAR POR LA PAREJA INTEGRANTES B Y C]
    
    Esta función debe:
    1. Leer el CSV de caras (faces_csv_path).
    2. Recorrer la lista de transcripciones (transcription_data).
    3. Para cada frase, filtrar el CSV por start_time y end_time.
    4. Calcular la MODA (emoción más frecuente) en ese rango.
    
    POR AHORA (MODO ARQUITECTO):
    Devuelve datos simulados (MOCK) para que el pipeline funcione.
    """
    
    log.info("--- INICIANDO SINCRONIZACIÓN (MODO MOCK/SIMULADO) ---")
    log.warning("AVISO: Usando lógica simulada. Los Integrantes B y C deben implementar la lógica real aquí.")
    
    # Validar que el CSV exista (aunque no lo usemos a fondo en el mock, es buena práctica)
    if not os.path.exists(faces_csv_path):
        log.error(f"No se encontró el CSV de caras: {faces_csv_path}")
        return []

    integrated_events = []

    # Iteramos sobre los datos reales del audio
    for segment in transcription_data:
        # --- ZONA DE TRABAJO PARA PAREJA B Y C ---
        # Aquí deberán leer el CSV real y calcular la moda.
        # Por ahora, inventamos que la cara siempre está "neutral".
        
        mock_face_emotion = "neutral (MOCK)" 
        mock_confidence = 0.99
        # -----------------------------------------

        event = {
            "start_time": segment['start_time'],
            "end_time": segment['end_time'],
            "text": segment['text'],
            "audio_emotion": "neutral", # Esto vendría del módulo de audio
            "face_emotion": mock_face_emotion, # <--- DATO SIMULADO
            "face_confidence": mock_confidence,
            "congruence": False # Asumimos falso por defecto
        }
        integrated_events.append(event)

    log.info(f"Sincronización simulada terminada. {len(integrated_events)} eventos procesados.")
    return integrated_events