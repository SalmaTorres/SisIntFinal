import os
import sys
import json
import torch

# --- CONFIGURACIÓN DE RUTAS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # 02_CODE
BASE = os.path.dirname(CURRENT_DIR) # SISINTFINAL
MODULES_PATH = os.path.join(CURRENT_DIR, "modules")

# Agregar 02_CODE al path para encontrar 'utils' y 'modules'
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

# --- PARÁMETROS ---
VIDEO_NAME = "video_05.mp4" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Rutas de Archivos
VIDEO_PATH = os.path.join(BASE, "01_DATA", "raw", VIDEO_NAME)
CLEAN_NAME = os.path.splitext(VIDEO_NAME)[0]
AUDIO_OUT = os.path.join(BASE, "01_DATA", "audio_clean", f"audio_{CLEAN_NAME}.wav")
CSV_OUT = os.path.join(BASE, "01_DATA", "series_temporales", f"{CLEAN_NAME}_faces.csv")
JSON_OUT = os.path.join(BASE, "05_OUTPUTS", "json_reports", f"{CLEAN_NAME}_FINAL.json")
IMG_OUT = os.path.join(BASE, "05_OUTPUTS", "visualizations", f"{CLEAN_NAME}_plot.png")

def run():
    log.info(f" INICIANDO PROCESAMIENTO: {VIDEO_NAME}")
    
    # 1. Validación inicial con tu helper
    if not validate_input_file(VIDEO_PATH):
        log.error(f"No se encuentra el video en: {VIDEO_PATH}")
        return

    # 2. Preparar Modelos
    ts.setup_pipelines(DEVICE)

    # 3. Audio y Texto
    transcription_data = []
    if ts.extract_audio(VIDEO_PATH, AUDIO_OUT):
        transcription_data = ts.get_transcription_and_emotion(AUDIO_OUT)
    else:
        return

    # 4. Facial (DeepFace)
    fe.extract_faces_from_video(VIDEO_PATH, CSV_OUT)

    # 5. Sincronización
    if transcription_data and os.path.exists(CSV_OUT):
        integrated_data = sy.synchronize_data(transcription_data, CSV_OUT)

        # 6. Guardar Reporte Final usando tu helper
        create_output_directory(os.path.dirname(JSON_OUT))
        with open(JSON_OUT, 'w', encoding='utf-8') as f:
            json.dump({"video": VIDEO_NAME, "events": integrated_data}, f, indent=4, ensure_ascii=False)
        log.info(f"Reporte JSON guardado: {JSON_OUT}")

        # 7. Gráfico
        vi.generate_comparison_plot(integrated_data, IMG_OUT)
        log.info(f"Gráfico guardado: {IMG_OUT}")
    else:
        log.error("Faltan datos para la sincronización final.")

    log.info("=== PIPELINE COMPLETADO ===")

if __name__ == "__main__":
    run()