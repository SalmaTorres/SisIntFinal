import os
import sys
import json
import torch
import time

# --- CONFIGURACIÓN DE RUTAS PARA IMPORTACIÓN ---
# CURRENT_DIR es SISINTFINAL/02_CODE
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) 
BASE = os.path.dirname(CURRENT_DIR) 

# Rutas según tu estructura de carpetas
MODULES_PATH = os.path.join(CURRENT_DIR, "modules")
UTILS_PATH = os.path.join(CURRENT_DIR, "utils")

# Agregar rutas al sistema para que Python encuentre los archivos .py
sys.path.append(CURRENT_DIR)
sys.path.append(MODULES_PATH)
sys.path.append(os.path.join(MODULES_PATH, "audio_text"))
sys.path.append(os.path.join(MODULES_PATH, "visual"))
sys.path.append(os.path.join(MODULES_PATH, "integration"))
sys.path.append(UTILS_PATH)

# --- IMPORTACIÓN DE TUS UTILS ---
try:
    # Se importa desde la carpeta utils
    from logger import get_logger
    from helpers import validate_input_file, create_output_directory
    log = get_logger("PIPELINE_PRINCIPAL")
except ImportError as e:
    print(f"Error crítico: No se encontraron tus utils en {UTILS_PATH}. Detalle: {e}")
    sys.exit(1)

# --- IMPORTACIÓN DE MÓDULOS ---
try:
    import transcriber as ts        # Desde modules/audio_text/transcriber.py
    import face_extractor as fe     # Desde modules/visual/face_extractor.py
    import synchronizer as sy       # Desde modules/integration/synchronizer.py
    import visualizer as vi         # Desde modules/integration/visualizer.py
    import analyzer as an           # Desde modules/integration/analyzer.py
    from validator import run_manual_validation # Desde modules/integration/validator.py
except ImportError as e:
    log.error(f"Error al importar módulos funcionales: {e}")
    sys.exit(1)

# --- PARÁMETROS GLOBALES ---
VIDEO_NAME = "video_04.mp4" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Definición de Rutas de Archivos (Estructura SISINTFINAL)
# Las entradas están en 01_DATA/raw
VIDEO_PATH = os.path.join(BASE, "01_DATA", "raw", VIDEO_NAME)
CLEAN_NAME = os.path.splitext(VIDEO_NAME)[0]

# Salidas Intermedias y Finales organizadas según tu estructura
AUDIO_OUT = os.path.join(BASE, "01_DATA", "audio_clean", f"audio_{CLEAN_NAME}.wav")
CSV_OUT = os.path.join(BASE, "01_DATA", "series_temporales", f"{CLEAN_NAME}_faces.csv")
JSON_OUT = os.path.join(BASE, "05_OUTPUTS", "json_reports", f"{CLEAN_NAME}_FINAL.json")
IMG_OUT = os.path.join(BASE, "05_OUTPUTS", "visualizations", f"{CLEAN_NAME}.png")

def run():
    start_time_pipeline = time.time()
    log.info(f"=== INICIANDO PIPELINE MULTIMODAL: {VIDEO_NAME} ===")
    
    # 1. Validación Inicial
    if not validate_input_file(VIDEO_PATH):
        log.error(f"Archivo de video no encontrado: {VIDEO_PATH}")
        return

    # 2. Inicializar Modelos de IA
    ts.setup_pipelines(DEVICE)

    # 3. FASE AUDIO/TEXTO
    transcription_data = []
    # Aseguramos que existan las carpetas de salida
    create_output_directory(os.path.dirname(AUDIO_OUT))
    
    if ts.extract_audio(VIDEO_PATH, AUDIO_OUT):
        # Según tu código, este método integra transcripción y emoción
        transcription_data = ts.get_transcription_and_emotion(AUDIO_OUT)
    else:
        log.error("Fallo en la extracción de audio. Abortando.")
        return

    # 4. FASE VISUAL (DeepFace)
    create_output_directory(os.path.dirname(CSV_OUT))
    fe.extract_faces_from_video(VIDEO_PATH, CSV_OUT, sample_rate=30)

    # 5. FASE DE SINCRONIZACIÓN E INTELIGENCIA (PBI 4.1, 4.2 & 4.3)
    if transcription_data and os.path.exists(CSV_OUT):
        log.info("Sincronizando fuentes y generando estructura de contrato...")
        
        # Obtiene los eventos integrados con el historial y nuevas llaves
        events = sy.synchronize_data(transcription_data, CSV_OUT)

        if not events:
            log.error("No se generaron eventos tras la sincronización.")
            return

        # 6. CÁLCULO DE MÉTRICAS GLOBALES (PBI 4.2)
        total_scores = [e['congruence_score'] for e in events if 'congruence_score' in e]
        overall_score = sum(total_scores) / len(total_scores) if total_scores else 0.0
        total_duration = events[-1]['end_time_sec'] if events else 0.0
        
        # 7. ENSAMBLAJE DEL REPORTE FINAL (Siguiendo Estructura de Contrato PBI 4.3)
        report_final = {
            "interview_id": f"INT-{CLEAN_NAME.upper()}-{int(time.time())}",
            "video_path": VIDEO_PATH,
            "global_metrics": {
                "overall_congruence_score": round(overall_score, 2),
                "total_duration_sec": round(total_duration, 2)
            },
            "events": events # Lista con transcribed_text, emotion_facial_history, etc.
        }

        # --- FASE DE INSIGHTS (PBI 4.3) ---
        log.info("Generando insights narrativos para el reporte...")
        for event in report_final["events"]:
            # analyzer.generate_insights evalúa el score y redacta la frase
            event["temporal_insight"] = an.generate_insights(event)

        # Guardar JSON final en 05_OUTPUTS
        create_output_directory(os.path.dirname(JSON_OUT))
        with open(JSON_OUT, 'w', encoding='utf-8') as f:
            json.dump(report_final, f, indent=4, ensure_ascii=False)
        
        log.info(f"Reporte Final guardado en: {JSON_OUT}")
        log.info(f"Métrica Overall de la Entrevista: {round(overall_score, 2)}")

        # 8. GENERACIÓN DE VISUALIZACIÓN
        create_output_directory(os.path.dirname(IMG_OUT))
        vi.generate_comparison_plot(events, IMG_OUT)
        log.info(f"Visualización guardada en: {IMG_OUT}")

        # 9. TCI4.6 - Validación de Robustez
        log.info("Ejecutando auditoría de métricas (TCI4.6)...")
        manual_csv = os.path.join(BASE, "01_DATA", "validation_labels.csv")
        
        if os.path.exists(manual_csv):
            robustness_report = run_manual_validation(JSON_OUT, manual_csv)
            if robustness_report:
                log.info(f"EL MODELO TIENE UNA PRECISIÓN DEL {robustness_report['robustness_accuracy']}% RESPECTO AL HUMANO.")
        else:
            log.warning(f"No se encontró archivo de validación manual en: {manual_csv}")

    else:
        log.error("No se pudo completar la sincronización. Verifica archivos intermedios.")

    duration = time.time() - start_time_pipeline
    log.info(f"=== PIPELINE FINALIZADO EN {duration:.2f} SEGUNDOS ===")

if __name__ == "__main__":
    run()