import sys
import ffmpeg 
import torch
import os
import json
from transformers import pipeline

# --- CONFIGURACIÓN GENERAL ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# TRUCO: Agregamos la carpeta 02_CODE al path de sistema para poder importar utils
sys.path.append(os.path.join(BASE_DIR, "02_CODE"))

# Ahora sí podemos importar el logger del proyecto
try:
    from utils.logger import get_logger
    from utils.helpers import validate_input_file, create_output_directory
    log = get_logger("Modulo_Audio_Texto")
except ImportError as e:
    print(f"ERROR CRÍTICO DE IMPORTACIÓN: {e}")
    sys.exit(1)
    
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# --- RUTAS DE ARCHIVOS ---
VIDEO_FILENAME = "video_03.mp4" # Asume un video de validación

video_name_clean = os.path.splitext(VIDEO_FILENAME)[0] # "video_03"
AUDIO_FILENAME = f"audio_extraido_{video_name_clean}.wav"

VIDEO_PATH = os.path.join(BASE_DIR, "01_DATA", "raw", VIDEO_FILENAME)
AUDIO_PATH = os.path.join(BASE_DIR, "01_DATA", "raw", AUDIO_FILENAME)

# --- ARCHIVOS DE SALIDA DINÁMICOS ---
JSON_FILENAME = f"{video_name_clean}.json"
OUTPUT_JSON_PATH = os.path.join(BASE_DIR, "05_OUTPUTS", "json_reports", "audio_text", JSON_FILENAME)

# --- MODELOS PREENTRENADOS ---
ASR_MODEL_NAME = "openai/whisper-small" 

# MODELO FUNCIONAL BERT ESPAÑOL (Cumple con la familia BERT de la rúbrica)
NLP_MODEL_NAME = "pysentimiento/robertuito-emotion-analysis"

# Inicializar pipelines globalmente
ASR_PIPELINE = None
NLP_PIPELINE = None

def setup_pipelines():
    """Inicializa los pipelines ASR y NLP al inicio del script."""
    global ASR_PIPELINE, NLP_PIPELINE
    log.info(f"Configurando modelos en dispositivo: {DEVICE}")

    try:
        log.info(f"Cargando modelo ASR ({ASR_MODEL_NAME})...")
        # Pipeline ASR (Whisper)
        ASR_PIPELINE = pipeline(
            "automatic-speech-recognition",
            model=ASR_MODEL_NAME,
            device=DEVICE,
            chunk_length_s=10
        )
        log.info("-> Pipeline ASR cargado exitosamente.")
    except Exception as e:
        log.error(f"Fallo crítico al cargar ASR: {e}")

    try:
        log.info(f"Cargando modelo NLP ({NLP_MODEL_NAME})...")
        # Pipeline NLP (Sentiment Analysis para emociones de texto, PBI 2.2)
        NLP_PIPELINE = pipeline(
            "text-classification",  # <--- CAMBIAR A CLASIFICACIÓN DE TEXTO
            model=NLP_MODEL_NAME, 
            tokenizer=NLP_MODEL_NAME,
            device=DEVICE,
            top_k=1
        )
        log.info("-> Pipeline NLP cargado exitosamente.")
    except Exception as e:
        log.error(f"Fallo crítico al cargar NLP: {e}")


# ==============================================================================
# TAREA PBI 2.3: Extracción de Audio del Video (Módulo Utilidades)
# ==============================================================================
def extract_audio(video_path: str, output_audio_path: str) -> bool:
    """Extrae el audio del video usando ffmpeg-python (Criterio 1 PBI 2.3)."""
    log.info(f"[PBI 2.3] Iniciando extracción de audio: {video_path}")
    # USO DEL HELPER: Validación de entrada
    if not validate_input_file(video_path):
        return False
        
    try:
        # **Lógica de ffmpeg-python**
        (
            ffmpeg
            .input(video_path)
            .output(output_audio_path, acodec='pcm_s16le', ar='16000', y='-y') 
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        log.info(f"Audio extraído y guardado en: {output_audio_path}")
        return True
    except Exception as e:
        log.error(f"Error extrayendo audio con ffmpeg: {e}")
        return False

# ==============================================================================
# FUNCIÓN INTERMEDIA: Transcripción (ASR)
# ==============================================================================
def get_transcription_with_timestamps(audio_path: str) -> list:
    """
    Realiza la transcripción ASR usando Whisper y extrae frases con timestamps.
    """
    if not ASR_PIPELINE:
        log.warning("ASR Pipeline no disponible. Saltando transcripción.")
        return []
    
    # USO DEL HELPER: Validar que el audio se creó bien
    if not validate_input_file(audio_path):
        return []
        
    log.info(f"[ASR] Procesando archivo de audio: {audio_path}")
    
    try:
        # generate_kwargs={"language": "spanish"} fuerza a Whisper a detectar español
        result = ASR_PIPELINE(audio_path, return_timestamps=True, generate_kwargs={"language": "spanish"})
        
        chunks = []
        if 'chunks' in result:
            for chunk in result['chunks']:
                text = chunk['text'].strip()
                if text:
                    chunks.append({
                        'start_time': chunk['timestamp'][0],
                        'end_time': chunk['timestamp'][1],
                        'text': text
                    })
        
        log.info(f"[ASR] Transcripción completada. {len(chunks)} segmentos detectados.")
        return chunks
    except Exception as e:
        log.error(f"Error durante la ejecución de ASR: {e}")
        return []

# ==============================================================================
# TAREA PBI 2.2: Implementación del Modelo NLP para Emociones
# ==============================================================================
def get_text_emotions(text_list: list) -> list:
    """
    Clasifica la emoción/sentimiento con el Transformer (PBI 2.2).
    """
    if not NLP_PIPELINE:
        log.warning("NLP Pipeline no disponible. Saltando análisis de emociones.")
        return []

    log.info(f"[PBI 2.2] Analizando emociones de {len(text_list)} segmentos de texto...")
    
    emotions = []
    
    try:
        results = NLP_PIPELINE(text_list)
        for res in results:
            # Manejo robusto de la salida del pipeline
            top = res[0] if isinstance(res, list) else res
            emotions.append({'label': top['label'], 'score': top['score']})
        
        log.info("[PBI 2.2] Análisis de emociones finalizado.")
        return emotions
    except Exception as e:
        log.error(f"Error durante el análisis NLP: {e}")
        # Fallback para no romper el pipeline
        return [{'label': 'neutral', 'score': 0.0} for _ in text_list]

# ==============================================================================
# TAREA PBI 2.5: Ensamblaje ASR/NLP a Salida JSON
# ==============================================================================
def assemble_asr_nlp_output(transcription_results: list) -> dict:
    """Ensambla el JSON final."""
    if not transcription_results: 
        log.warning("No hay resultados de transcripción para guardar.")
        return {}

    text_list = [t['text'] for t in transcription_results]
    emotions = get_text_emotions(text_list)
    
    final_output = {
        'audio_analysis': {
            'transcribed_text': [],
            'text_emotions': []
        },
        'metadata': {
            'module': 'Audio/Texto',
            'video_source': VIDEO_FILENAME,
            'asr_model': ASR_MODEL_NAME,
            'nlp_model': NLP_MODEL_NAME
        }
    }

    log.info("Ensamblando estructura JSON final...")
    for i, t in enumerate(transcription_results):
        if i >= len(emotions): break
        
        # Corrección de timestamps nulos
        start = t['start_time'] if t['start_time'] is not None else 0.0
        # Si end_time es None, asumimos +2 segundos para evitar errores
        end = t['end_time'] if t['end_time'] is not None else start + 2.0
        
        # Agregar a sección transcripción
        final_output['audio_analysis']['transcribed_text'].append({
            'start_time': start,
            'end_time': end,
            'speaker': 'Sujeto',
            'text': t['text']
        })
        
        # Agregar a sección emociones
        final_output['audio_analysis']['text_emotions'].append({
            'start_time': start,
            'end_time': end,
            'emotion': emotions[i]['label'],
            'confidence': emotions[i]['score']
        })

    # USO DEL HELPER: Crear directorio de salida si no existe
    create_output_directory(os.path.dirname(OUTPUT_JSON_PATH))
    
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
        
    log.info(f"[PBI 2.5] Reporte JSON guardado exitosamente en: {OUTPUT_JSON_PATH}")
    return final_output

# ==============================================================================
# FUNCIÓN PRINCIPAL DE EJECUCIÓN DEL MÓDULO (Día 2 Entregable)
# ==============================================================================
def main_module_run():
    """Ejecuta la secuencia completa del módulo Audio/Texto."""
    log.info("=== INICIANDO MÓDULO AUDIO/TEXTO ===")
    
    # 1. Configuración de pipelines (necesario para todas las tareas)
    setup_pipelines()
    
    if extract_audio(VIDEO_PATH, AUDIO_PATH):
        results = get_transcription_with_timestamps(AUDIO_PATH)
        assemble_asr_nlp_output(results)
    else:
        log.error("No se pudo completar el flujo por error en la extracción de audio.")
    
    log.info("=== PROCESO FINALIZADO ===")

if __name__ == "__main__":
    main_module_run()