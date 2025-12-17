import os
import sys
import json
import time

# --- CONFIGURACIÓN DE PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Importamos Módulos (Asegúrate de que los nombres de archivo coincidan)
from utils.logger import get_logger
from utils.helpers import validate_input_file, create_output_directory
from modules.audio_text.transcriber import setup_pipelines, extract_audio, get_transcription_with_timestamps, assemble_asr_nlp_output
from modules.visual.face_extractor import extract_faces_from_video
from modules.integration.synchronizer import synchronize_multimodal_data

# Configuración Global
log = get_logger("PIPELINE_PRINCIPAL")
ROOT_DIR = os.path.dirname(BASE_DIR) # SISINTFINAL

# --- PARÁMETROS DE ENTRADA (CAMBIAR AQUÍ EL VIDEO A PROCESAR) ---
VIDEO_NAME = "video_01.mp4" 
VIDEO_PATH = os.path.join(ROOT_DIR, "01_DATA", "raw", VIDEO_NAME)

def run_full_pipeline():
    start_global = time.time()
    log.info(f" INICIANDO PIPELINE END-TO-END: {VIDEO_NAME}")

    setup_pipelines() 

    # 1. Validaciones Iniciales
    if not validate_input_file(VIDEO_PATH):
        log.error("Abortando pipeline: No se encuentra el video de entrada.")
        return

    # Preparar nombres de archivos intermedios
    filename_clean = os.path.splitext(VIDEO_NAME)[0]
    audio_path = os.path.join(ROOT_DIR, "01_DATA", "processed", "audio_clean", f"audio_{filename_clean}.wav")
    faces_csv_path = os.path.join(ROOT_DIR, "05_OUTPUTS", "series_temporales", f"{filename_clean}_faces.csv")
    final_json_path = os.path.join(ROOT_DIR, "05_OUTPUTS", "final_integration", f"{filename_clean}_FINAL.json")

    # ---------------------------------------------------------
    # FASE 1: PROCESAMIENTO DE AUDIO / TEXTO
    # ---------------------------------------------------------
    log.info(">>> FASE 1: Audio y NLP")
    if extract_audio(VIDEO_PATH, audio_path):
        # Obtenemos la lista de transcripciones (con timestamps)
        transcription_results = get_transcription_with_timestamps(audio_path)
        # Generamos el reporte individual de audio (Opcional, pero bueno para debug)
        audio_json = assemble_asr_nlp_output(transcription_results) 
        
        # Extraemos la data enriquecida de audio (con emociones de texto si ya las calculaste)
        # Nota: Para el PBI 3.1 usaremos 'transcription_results' y el CSV
    else:
        log.error("Fallo en extracción de audio.")
        return

    # ---------------------------------------------------------
    # FASE 2: PROCESAMIENTO VISUAL (DeepFace)
    # ---------------------------------------------------------
    log.info(">>> FASE 2: Extracción Facial (CNN)")
    # Este script genera el CSV en la ruta faces_csv_path
    # sample_rate=15 para ir rápido en pruebas, bajar a 5 o 1 para prod
    extract_faces_from_video(VIDEO_PATH, sample_rate=15) 

    if not os.path.exists(faces_csv_path):
        log.error("Fallo crítico: No se generó el CSV de caras.")
        return

    # ---------------------------------------------------------
    # FASE 3: INTEGRACIÓN Y SINCRONIZACIÓN
    # ---------------------------------------------------------
    log.info(">>> FASE 3: Sincronización Multimodal")
    
    # Llamamos a la función (que ahora es MOCK, luego será REAL)
    integrated_events = synchronize_multimodal_data(transcription_results, faces_csv_path)

    # ---------------------------------------------------------
    # FASE 4: GENERACIÓN DEL ENTREGABLE FINAL
    # ---------------------------------------------------------
    log.info(">>> FASE 4: Guardado de Resultados (Contract Compliance)")
    
    # 1. Calcular Métricas Globales (Promedio de congruencia)
    total_congruence = sum([e['congruence_score'] for e in integrated_events])
    count_events = len(integrated_events)
    avg_congruence = round(total_congruence / count_events, 2) if count_events > 0 else 0.0
    
    # 2. Obtener duración total (del último evento)
    total_duration = integrated_events[-1]['end_time_sec'] if integrated_events else 0.0

    # 3. Construir el JSON EXACTO según tu output_structure_contract.json
    final_output = {
        "interview_id": f"INT-{filename_clean}", # Generamos un ID basado en el nombre
        "video_path": VIDEO_PATH,
        "global_metrics": {
            "overall_congruence_score": avg_congruence,
            "total_duration_sec": total_duration
        },
        "events": integrated_events # Esta lista viene del synchronizer.py
    }

    create_output_directory(os.path.dirname(final_json_path))
    with open(final_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    log.info(f" Archivo Final: {final_json_path}")

if __name__ == "__main__":
    run_full_pipeline()