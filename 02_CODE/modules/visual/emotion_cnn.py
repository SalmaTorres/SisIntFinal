import json
from collections import Counter
import os
import pandas as pd
import sys

# --- AJUSTE DE IMPORTACIONES ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", ".."))

from utils.logger import get_logger
from utils.helpers import validate_input_file, create_output_directory

# --- CONFIGURACIÓN ---
log = get_logger("CNN_Consolidacion")

# --- CONFIGURACIÓN DE RUTAS ---
# Obtenemos la ruta base del script (02_CODE)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
ROOT_DIR = os.path.dirname(BASE_DIR)

# Cambia esto manualmente según el video que estés probando hoy
NOMBRE_VIDEO = "video_03" 

INPUT_TIMESERIES_PATH_CSV = os.path.join(ROOT_DIR, "05_OUTPUTS", "series_temporales", f"{NOMBRE_VIDEO}_faces.csv")
INPUT_AUDIO_TEXT_PATH = os.path.join(ROOT_DIR, "05_OUTPUTS", "json_reports", "audio_text", f"{NOMBRE_VIDEO}.json")
OUTPUT_CNN_PATH = os.path.join(ROOT_DIR, "05_OUTPUTS", "json_reports", "face_analysis", f"{NOMBRE_VIDEO}.json")

# --- LÓGICA CENTRAL PBI 2.4 ---

def consolidate_emotions_by_segment(timeseries_data_list: list, start_time: float, end_time: float) -> dict:
    """
    Implementa la lógica de votación (Moda) para un segmento de tiempo. (Criterio 2)
    """
    
    # 1. Filtrar los frames del segmento de tiempo
    segment_frames = [
        frame for frame in timeseries_data_list 
        if start_time <= frame['timestamp_sec'] < end_time
    ]
    
    if not segment_frames:
        # Si no hay frames, asumimos neutralidad
        return {
            'emotion_facial_mode': 'neutral', 
            'confidence_facial_mode': 0.0, 
            'emotion_facial_history': []
        }

    # 2. Votación: Contar frecuencias de emociones
    emotion_counts = Counter([frame['emotion'] for frame in segment_frames])
    
    # 3. Determinar la emoción dominante (la Moda)
    dominant_emotion, count = emotion_counts.most_common(1)[0]
    
    # 4. Calcular Confianza Simplificada: Porcentaje de frames que votaron por la moda
    total_frames = len(segment_frames)
    average_confidence_simplified = count / total_frames if total_frames > 0 else 0.0

    log.info(f"Segmento [{start_time:.2f}s - {end_time:.2f}s] -> Dominante: {dominant_emotion.upper()} ({count}/{total_frames} frames)")
    
    return {
        # Campos requeridos para la integración (Día 3)
        'start_time_sec': start_time,
        'end_time_sec': end_time,
        'emotion_facial_mode': dominant_emotion,
        'confidence_facial_mode': average_confidence_simplified,
        # Historia requerida para Análisis Temporal Avanzado (PBI 4.1)
        'emotion_facial_history': [frame['emotion'] for frame in segment_frames] 
    }

# --- FUNCIÓN PRINCIPAL DE EJECUCIÓN DEL MÓDULO ---

def main_cnn_module_run():
    """Ejecuta el PBI 2.4 completo."""
    log.info("Iniciando Consolidación de Emociones (PBI 2.4)...")
    
    # Criterio 1: Leer la serie temporal (CSV)
    # 1. Validación de Inputs con Helper
    if not validate_input_file(INPUT_TIMESERIES_PATH_CSV):
        log.error("Falta el archivo CSV de series temporales. Ejecuta primero 'face_extractor.py'.")
        return
        
    if not validate_input_file(INPUT_AUDIO_TEXT_PATH):
        log.error("Falta el archivo JSON del módulo de Audio. Ejecuta primero 'transcriber.py'.")
        return
        
    # 2. Carga de Datos
    try:
        df_timeseries = pd.read_csv(INPUT_TIMESERIES_PATH_CSV)
        timeseries_data_list = df_timeseries.to_dict('records')
        log.info(f"Serie temporal cargada: {len(timeseries_data_list)} registros.")
        
        with open(INPUT_AUDIO_TEXT_PATH, 'r', encoding='utf-8') as f:
            audio_text_data = json.load(f)
        segments = audio_text_data['audio_analysis']['transcribed_text']
        log.info(f"Segmentos de transcripción cargados: {len(segments)} segmentos.")
        
    except Exception as e:
        log.error(f"Error leyendo archivos de entrada: {e}")
        return
        
    # 3. Procesamiento (Consolidación)
    consolidated_results = []
    for segment in segments:
        consolidated_data = consolidate_emotions_by_segment(
            timeseries_data_list, 
            segment['start_time'], 
            segment['end_time']
        )
        consolidated_results.append(consolidated_data)
    
    # 4. Generación de Salida
    output_data = {
        'face_analysis': {
            'consolidated_segments': consolidated_results,
            'metadata': {
                'processing_module': 'CNN/Emociones',
                'consolidation_method': 'Moda (Votación)'
            }
        }
    }

    try:
        create_output_directory(os.path.dirname(OUTPUT_CNN_PATH))
        with open(OUTPUT_CNN_PATH, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        
        log.info(f"¡Éxito! Salida consolidada guardada en: {OUTPUT_CNN_PATH}")
        
    except Exception as e:
        log.error(f"Error guardando el JSON de salida: {e}")

if __name__ == "__main__":
    main_cnn_module_run()