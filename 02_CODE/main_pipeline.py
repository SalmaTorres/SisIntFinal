import os
import sys
import json
import torch
import time

# --- CONFIGURACIÓN DE RUTAS PARA IMPORTACIÓN ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # 02_CODE
BASE = os.path.dirname(CURRENT_DIR) # SISINTFINAL
MODULES_PATH = os.path.join(CURRENT_DIR, "modules")

# Agregar rutas al sistema para encontrar utils y módulos
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.join(MODULES_PATH, "audio_text"))
sys.path.append(os.path.join(MODULES_PATH, "visual"))
sys.path.append(os.path.join(MODULES_PATH, "integration"))

# --- IMPORTACIÓN DE TUS UTILS ---
try:
    from utils.logger import get_logger
    from utils.helpers import validate_input_file, create_output_directory
    log = get_logger("PIPELINE_PRINCIPAL")
except ImportError as e:
    print(f"Error crítico: No se encontraron tus utils. {e}")
    sys.exit(1)

# Importación de Módulos
import transcriber as ts
import face_extractor as fe
import synchronizer as sy
import visualizer as vi

# --- PARÁMETROS GLOBALES ---
VIDEO_NAME = "video_06.mp4" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Definición de Rutas de Archivos (Estructura SISINTFINAL)
VIDEO_PATH = os.path.join(BASE, "01_DATA", "raw", VIDEO_NAME)
CLEAN_NAME = os.path.splitext(VIDEO_NAME)[0]

# Salidas Intermedias y Finales
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
    if ts.extract_audio(VIDEO_PATH, AUDIO_OUT):
        transcription_data = ts.get_transcription_and_emotion(AUDIO_OUT)
    else:
        log.error("Fallo en la extracción de audio. Abortando.")
        return

    # 4. FASE VISUAL (DeepFace)
    fe.extract_faces_from_video(VIDEO_PATH, CSV_OUT, sample_rate=15)

    # 5. FASE DE SINCRONIZACIÓN E INTELIGENCIA (PBI 4.1, 4.2 & 4.3)
    if transcription_data and os.path.exists(CSV_OUT):
        log.info("Sincronizando fuentes y generando estructura de contrato...")
        
        # Obtiene los eventos integrados con el historial y nuevas llaves
        events = sy.synchronize_data(transcription_data, CSV_OUT)

        if not events:
            log.error("No se generaron eventos tras la sincronización.")
            return

        # 6. CÁLCULO DE MÉTRICAS GLOBALES (PBI 4.2)
        total_scores = [e['congruence_score'] for e in events]
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

        # Guardar JSON usando tu helper
        create_output_directory(os.path.dirname(JSON_OUT))
        with open(JSON_OUT, 'w', encoding='utf-8') as f:
            json.dump(report_final, f, indent=4, ensure_ascii=False)
        
        log.info(f"Reporte Final guardado en: {JSON_OUT}")
        log.info(f"Métrica Overall de la Entrevista: {round(overall_score, 2)}")

        # 8. GENERACIÓN DE VISUALIZACIÓN
        create_output_directory(os.path.dirname(IMG_OUT))
        vi.generate_comparison_plot(events, IMG_OUT)
        log.info(f"Visualización guardada en: {IMG_OUT}")

    else:
        log.error("No se pudo completar la sincronización. Verifica archivos intermedios.")

    duration = time.time() - start_time_pipeline
    log.info(f"=== PIPELINE FINALIZADO EN {duration:.2f} SEGUNDOS ===")

if __name__ == "__main__":
    run()