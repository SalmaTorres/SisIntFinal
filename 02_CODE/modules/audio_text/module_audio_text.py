import sys
import ffmpeg 
import torch
import os
import json
import nltk
from transformers import pipeline

# --- CONFIGURACIÓN DE RUTAS E IMPORTACIONES ---
# Ajuste para importar desde utils
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(BASE_DIR, "02_CODE"))

try:
    from utils.logger import get_logger
    from utils.helpers import validate_input_file, create_output_directory
    log = get_logger("Modulo_Audio_Texto")
except ImportError as e:
    print(f"ERROR CRÍTICO: No se encontraron los módulos utils. {e}")
    sys.exit(1)

# Descarga de recursos NLTK (Aporte del código de tu amiga para robustez)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    log.warning(f"No se pudo descargar NLTK completo, se usará tokenización básica: {e}")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# --- CONFIGURACIÓN DE ARCHIVOS (DINÁMICO) ---
# CAMBIA ESTO POR EL VIDEO QUE QUIERAS PROBAR
VIDEO_FILENAME = "video_04.mp4" 

# Generación automática de nombres
video_name_clean = os.path.splitext(VIDEO_FILENAME)[0]
AUDIO_FILENAME = f"audio_extraido_{video_name_clean}.wav"
JSON_FILENAME = f"{video_name_clean}.json"

# Rutas completas
VIDEO_PATH = os.path.join(BASE_DIR, "01_DATA", "raw", VIDEO_FILENAME)
AUDIO_PATH = os.path.join(BASE_DIR, "01_DATA", "processed", "audio_clean", AUDIO_FILENAME)
OUTPUT_JSON_PATH = os.path.join(BASE_DIR, "05_OUTPUTS", "json_reports", "audio_text", JSON_FILENAME)

# --- MODELOS ---
ASR_MODEL_NAME = "openai/whisper-small" 
NLP_MODEL_NAME = "pysentimiento/robertuito-emotion-analysis" # Modelo específico de emociones

ASR_PIPELINE = None
NLP_PIPELINE = None

def setup_pipelines():
    """Inicializa los modelos con manejo de errores."""
    global ASR_PIPELINE, NLP_PIPELINE
    log.info(f"Configurando modelos en dispositivo: {DEVICE}")

    try:
        log.info(f"Cargando ASR ({ASR_MODEL_NAME})...")
        ASR_PIPELINE = pipeline(
            "automatic-speech-recognition",
            model=ASR_MODEL_NAME,
            device=DEVICE,
            chunk_length_s=30
        )
    except Exception as e:
        log.error(f"Error cargando ASR: {e}")

    try:
        log.info(f"Cargando NLP ({NLP_MODEL_NAME})...")
        NLP_PIPELINE = pipeline(
            "text-classification", 
            model=NLP_MODEL_NAME, 
            tokenizer=NLP_MODEL_NAME,
            device=DEVICE,
            top_k=1
        )
    except Exception as e:
        log.error(f"Error cargando NLP: {e}")

# ==============================================================================
# 1. EXTRACCIÓN DE AUDIO (Usa Helpers + ffmpeg)
# ==============================================================================
def extract_audio(video_path: str, output_audio_path: str) -> bool:
    log.info(f"Iniciando extracción de audio desde: {os.path.basename(video_path)}")
    
    if not validate_input_file(video_path):
        return False
        
    # Crear carpeta de destino si no existe
    create_output_directory(os.path.dirname(output_audio_path))

    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_audio_path, acodec='pcm_s16le', ar='16000', y='-y') 
            .run(capture_stdout=True, capture_stderr=True)
        )
        log.info(f"Audio guardado en: {output_audio_path}")
        return True
    except Exception as e:
        log.error(f"Error en ffmpeg: {e}")
        return False

# ==============================================================================
# 2. TRANSCRIPCIÓN (ASR)
# ==============================================================================
def get_transcription_with_timestamps(audio_path: str) -> list:
    if not ASR_PIPELINE:
        log.error("Pipeline ASR no inicializado.")
        return []
    
    if not validate_input_file(audio_path):
        return []
        
    log.info("Iniciando transcripción...")
    try:
        # generate_kwargs ayuda a forzar el idioma si es necesario
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
        
        log.info(f"Transcripción finalizada: {len(chunks)} segmentos.")
        return chunks
    except Exception as e:
        log.error(f"Fallo en ASR: {e}")
        return []

# ==============================================================================
# 3. ANÁLISIS DE EMOCIONES (INTEGRACIÓN LÓGICA DE TU AMIGA)
# ==============================================================================
def get_text_emotions(text_list: list) -> list:
    """
    Procesa frase por frase (Iterativo) para mayor seguridad.
    Esta es la mejora clave tomada del código de tu amiga.
    """
    if not NLP_PIPELINE:
        log.warning("NLP no disponible. Retornando neutros.")
        return [{'label': 'neutral', 'score': 0.0} for _ in text_list]

    log.info(f"Analizando emociones de {len(text_list)} frases...")
    emotions = []

    # --- LÓGICA ROBUSTA (FRIEND'S VERSION) ---
    for i, text in enumerate(text_list):
        if not text.strip():
            emotions.append({'label': 'neutral', 'score': 0.0})
            continue

        try:
            # Procesamos INDIVIDUALMENTE
            res = NLP_PIPELINE(text)
            
            # Normalizamos la salida (a veces es lista de listas)
            top = res[0] if isinstance(res, list) else res
            
            emotions.append({
                'label': top['label'], 
                'score': top['score']
            })
            
            # Logueo detallado (opcional, bueno para debug)
            # log.info(f"Frase {i}: {top['label']} ({top['score']:.2f})")
            
        except Exception as e:
            log.warning(f"Error analizando frase {i}: '{text}'. Usando fallback Neutral.")
            emotions.append({'label': 'neutral', 'score': 0.0})
    
    return emotions

# ==============================================================================
# 4. ENSAMBLAJE FINAL
# ==============================================================================
def assemble_asr_nlp_output(transcription_results: list) -> dict:
    if not transcription_results:
        return {}

    # Extraemos textos
    text_list = [t['text'] for t in transcription_results]
    
    # Obtenemos emociones (usando la función mejorada)
    emotions = get_text_emotions(text_list)
    
    final_output = {
        'audio_analysis': {
            'transcribed_text': [],
            'text_emotions': []
        },
        'metadata': {
            'processing_module': 'Audio/Texto',
            'video_source': VIDEO_FILENAME,
            'asr_model': ASR_MODEL_NAME,
            'nlp_model': NLP_MODEL_NAME
        }
    }

    # Unimos todo respetando los timestamps originales
    for i, t in enumerate(transcription_results):
        if i >= len(emotions): break
        
        start = t['start_time'] if t['start_time'] is not None else 0.0
        end = t['end_time'] if t['end_time'] is not None else start + 2.0
        
        # Transcripción
        final_output['audio_analysis']['transcribed_text'].append({
            'start_time': start,
            'end_time': end,
            'speaker': 'Sujeto',
            'text': t['text']
        })
        
        # Emoción
        final_output['audio_analysis']['text_emotions'].append({
            'start_time': start,
            'end_time': end,
            'emotion': emotions[i]['label'],
            'confidence': emotions[i]['score']
        })

    create_output_directory(os.path.dirname(OUTPUT_JSON_PATH))
    
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
        
    log.info(f"Reporte generado: {OUTPUT_JSON_PATH}")
    return final_output

# ==============================================================================
# MAIN
# ==============================================================================
def main_module_run():
    setup_pipelines()
    
    if extract_audio(VIDEO_PATH, AUDIO_PATH):
        results = get_transcription_with_timestamps(AUDIO_PATH)
        assemble_asr_nlp_output(results)
    else:
        log.error("Proceso abortado por fallo en audio.")

if __name__ == "__main__":
    main_module_run()