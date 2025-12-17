from utils.logger import get_logger

# Inicializar logger con el nombre del módulo
log = get_logger("Modulo_CNN_Emociones")

def analizar_video(path):
    log.info(f"Iniciando análisis facial del video: {path}")
    # ... tu código ...
    log.info("Análisis facial completado con éxito.")
    
import json
from collections import Counter
import os
import pandas as pd

# --- CONFIGURACIÓN DE RUTAS ---
# Obtenemos la ruta base del script (02_CODE)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# INPUT 1 (Criterio 1): Serie temporal facial (Salida de PBI 2.1)
INPUT_TIMESERIES_PATH_CSV = os.path.join(BASE_DIR, "..", "04_OUTPUTS", "cnn_time_series.csv")

# INPUT 2: Segmentos de audio/texto (Salida de PBI 2.3)
INPUT_AUDIO_TEXT_PATH = os.path.join(BASE_DIR, "..", "04_OUTPUTS", "audio_text_module_output.json")

# OUTPUT (Criterio 3): Salida consolidada del módulo CNN
OUTPUT_CNN_PATH = os.path.join(BASE_DIR, "..", "04_OUTPUTS", "cnn_module_output.json")

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

    print(f"\n[PBI 2.4] Segmento [{start_time:.2f}s - {end_time:.2f}s]: Emoción Dominante: {dominant_emotion.upper()}")
    
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
    
    # Criterio 1: Leer la serie temporal (CSV)
    try:
        df_timeseries = pd.read_csv(INPUT_TIMESERIES_PATH_CSV)
        # Convertir a lista de diccionarios para procesamiento eficiente
        timeseries_data_list = df_timeseries.to_dict('records')
        print(f"[PBI 2.4] Criterio 1 OK. Serie temporal leída ({len(timeseries_data_list)} frames).")
    except FileNotFoundError:
        print(f"ERROR CRÍTICO: Archivo de serie temporal no encontrado en {INPUT_TIMESERIES_PATH_CSV}.")
        print("Asegúrate de ejecutar el código del PBI 2.1 para generar el CSV primero.")
        return
        
    # Obtener los límites de tiempo de los segmentos de la transcripción
    try:
        with open(INPUT_AUDIO_TEXT_PATH, 'r') as f:
            audio_text_data = json.load(f)
        segments = audio_text_data['audio_analysis']['transcribed_text']
    except FileNotFoundError:
        print(f"ERROR CRÍTICO: No se encontró la salida del módulo Audio/Texto.")
        return
        
    # Criterio 2: Consolidar cada segmento usando la lógica de votación
    consolidated_results = []
    for segment in segments:
        consolidated_data = consolidate_emotions_by_segment(
            timeseries_data_list, 
            segment['start_time'], 
            segment['end_time']
        )
        consolidated_results.append(consolidated_data)
    
    # Criterio 3: Generar el archivo JSON temporal de salida
    output_data = {
        'face_analysis': {
            'consolidated_segments': consolidated_results,
            'metadata': {
                'processing_module': 'CNN/Emociones',
                'consolidation_method': 'Moda (Votación)'
            }
        }
    }

    with open(OUTPUT_CNN_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
        
    print(f"\n[PBI 2.4] Criterio 3 OK. Salida del Módulo CNN generada en {OUTPUT_CNN_PATH}.")
    print("\n¡PBI 2.4 COMPLETADO! Módulo CNN/Emociones listo para la integración.")

if __name__ == "__main__":
    main_cnn_module_run()