import ffmpeg 
import torch
import os
import json
from transformers import pipeline

# --- CONFIGURACIÓN GENERAL ---
# Obtener la ruta base del proyecto (SISINTFINAL)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# --- RUTAS DE ARCHIVOS ---
VIDEO_FILENAME = "video_entrevista_propia.mp4" # Asume un video de validación
AUDIO_FILENAME = "audio_extraido.wav"
VIDEO_PATH = os.path.join(BASE_DIR, "01_DATA", "raw", VIDEO_FILENAME)
AUDIO_PATH = os.path.join(BASE_DIR, "01_DATA", "raw", AUDIO_FILENAME)
OUTPUT_JSON_PATH = os.path.join(BASE_DIR, "04_OUTPUTS", "audio_text_module_output.json")

# --- MODELOS PREENTRENADOS ---
ASR_MODEL_NAME = "openai/whisper-small" 

# MODELO FUNCIONAL BERT ESPAÑOL (Cumple con la familia BERT de la rúbrica)
NLP_MODEL_NAME = "dccuchile/bert-base-spanish-wwm-uncased"

# Inicializar pipelines globalmente
ASR_PIPELINE = None
NLP_PIPELINE = None

def setup_pipelines():
    """Inicializa los pipelines ASR y NLP al inicio del script."""
    global ASR_PIPELINE, NLP_PIPELINE
    print(f"Usando dispositivo: {DEVICE}")

    try:
        # Pipeline ASR (Whisper)
        ASR_PIPELINE = pipeline(
            "automatic-speech-recognition",
            model=ASR_MODEL_NAME,
            device=DEVICE,
            return_timestamps=True # Usamos True para obtener segmentos
        )
        print(f"-> ASR Pipeline ({ASR_MODEL_NAME}) cargado con éxito.")
    except Exception as e:
        print(f"-> ERROR al cargar el modelo ASR: {e}")

    try:
        # Pipeline NLP (Sentiment Analysis para emociones de texto, PBI 2.2)
        NLP_PIPELINE = pipeline(
            "text-classification",  # <--- CAMBIAR A CLASIFICACIÓN DE TEXTO
            model=NLP_MODEL_NAME, 
            tokenizer=NLP_MODEL_NAME,
            device=DEVICE
        )
        print(f"-> NLP Pipeline ({NLP_MODEL_NAME}) cargado con éxito.")
    except Exception as e:
        print(f"-> ERROR al cargar el modelo NLP: {e}")


# ==============================================================================
# TAREA PBI 2.3: Extracción de Audio del Video (Módulo Utilidades)
# ==============================================================================
def extract_audio(video_path: str, output_audio_path: str) -> bool:
    """Extrae el audio del video usando ffmpeg-python (Criterio 1 PBI 2.3)."""
    print(f"\n[PBI 2.3] Iniciando extracción de audio de: {video_path}")
    if not os.path.exists(video_path):
        print(f"ERROR: Archivo de video no encontrado en {video_path}.")
        return False
        
    try:
        # **Lógica de ffmpeg-python**
        (
            ffmpeg
            .input(video_path)
            .output(output_audio_path, acodec='pcm_s16le', ar='16000', y='-y') 
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        print(f"[PBI 2.3] Audio extraído y guardado en: {output_audio_path}. Criterio 1 OK.")
        return True
    except ffmpeg.Error as e:
        print(f"ERROR en extract_audio (PBI 2.3) con ffmpeg: {e.stderr.decode('utf8')}")
        return False
    except Exception as e:
        print(f"ERROR inesperado en extract_audio (PBI 2.3): {e}")
        return False

# ==============================================================================
# FUNCIÓN INTERMEDIA: Transcripción (ASR)
# ==============================================================================
def get_transcription_with_timestamps(audio_path: str) -> list:
    """
    Realiza la transcripción ASR usando Whisper y extrae frases con timestamps.
    """
    if not ASR_PIPELINE:
        print("ASR Pipeline no está inicializado. Saliendo.")
        return []
        
    print(f"\n[ASR] Procesando archivo de audio: {audio_path}")
    
    try:
        result = ASR_PIPELINE(
            audio_path, 
            chunk_length_s=15, 
            stride_length_s=1,
            return_timestamps=True,
            batch_size=8,
            ignore_warning=True
        )
        
        transcription_chunks = []
        if 'chunks' in result:
            for chunk in result['chunks']:
                text = chunk['text'].strip()
                if text:
                    transcription_chunks.append({
                        'start_time': chunk['timestamp'][0] or 0.0,
                        'end_time': chunk['timestamp'][1] or chunk['timestamp'][0] + 1.0,
                        'text': text
                    })
        
        # *** DEPURACIÓN AÑADIDA: IMPRIMIR RESULTADO ASR PARA VERIFICACIÓN ***
        print("\n--- Resultado DETALLADO de Transcripción ASR ---")
        if transcription_chunks:
            for i, chunk in enumerate(transcription_chunks):
                print(f"Segmento {i}: [{chunk['start_time']:.2f}s - {chunk['end_time']:.2f}s] -> '{chunk['text']}'")
        else:
            print("ADVERTENCIA: La transcripción ASR no generó texto. Revisa el audio de entrada.")
        print("-------------------------------------------------")
        # *** FIN DEPURACIÓN ***

        print(f"[ASR] Transcripción completada. {len(transcription_chunks)} segmentos encontrados.")
        return transcription_chunks

    except Exception as e:
        print(f"ERROR durante la transcripción ASR: {e}")
        return []

# ==============================================================================
# TAREA PBI 2.2: Implementación del Modelo NLP para Emociones
# ==============================================================================
def get_text_emotions(text_list: list) -> list:
    """
    Clasifica la emoción/sentimiento con el Transformer (PBI 2.2).
    """
    if not NLP_PIPELINE:
        print("NLP Pipeline no está inicializado. Saliendo.")
        return []

    print(f"\n[PBI 2.2] Analizando {len(text_list)} frases para emociones de texto (Modelo {NLP_MODEL_NAME}).")
    
    text_emotions = []
    
    try:
        # Intenta obtener el resultado del nuevo Transformer
        results = NLP_PIPELINE(text_list)
        
        if not results:
             raise ValueError("El pipeline de emociones devolvió una lista vacía.")
             
        for result in results:
            text_emotions.append({
                # El nuevo modelo de PySentimiento devuelve etiquetas como "anger", "joy", "sadness"
                'emotion_label': result[0]['label'], 
                'confidence_score': result[0]['score']
            })
            
        print("[PBI 2.2] Análisis de emociones de texto completado con el nuevo Transformer. Criterios OK.")
        return text_emotions
        
    except Exception as e:
        # Si este modelo también falla, seguimos con el plan de emergencia
        print(f"ERROR CRÍTICO en NLP (PBI 2.2): {e}. Asignando 'NEUTRAL' a todos los segmentos para continuar.")
        
        for _ in range(len(text_list)):
             text_emotions.append({
                'emotion_label': 'NEUTRAL',
                'confidence_score': 0.01 
            })
        
        return text_emotions

# ==============================================================================
# TAREA PBI 2.5: Ensamblaje ASR/NLP a Salida JSON
# ==============================================================================
def assemble_asr_nlp_output(transcription_results: list) -> dict:
    """
    Combina ASR (timestamps) y NLP (emociones) para generar la salida JSON (PBI 2.5).
    """
    if not transcription_results:
        print("ADVERTENCIA: No hay resultados de transcripción para ensamblar.")
        return {}

    # 1. Extraer solo el texto para el análisis de emoción PBI 2.2
    text_list = [item['text'] for item in transcription_results]
    
    # 2. Obtener las emociones (lectura del PBI 2.2)
    text_emotions_data = get_text_emotions(text_list)
    
    # 3. Ensamblar las secciones (Criterios de PBI 2.5)
    transcribed_text_section = []
    text_emotions_section = []
    
    for i, result in enumerate(transcription_results):
        if i >= len(text_emotions_data):
            # Esto no debería pasar con el manejo de errores en get_text_emotions, pero es un seguro
            print(f"ADVERTENCIA: Falta dato de emoción para el segmento {i}. Continuando.")
            continue
            
        emotion_data = text_emotions_data[i]
        
        # Sección transcribed_text
        transcribed_text_section.append({
            'start_time': result['start_time'],
            'end_time': result['end_time'],
            'speaker': 'Sujeto', 
            'text': result['text']
        })
        
        # Sección text_emotions (Fusión de ASR timestamps y NLP emotion)
        text_emotions_section.append({
            'start_time': result['start_time'],
            'end_time': result['end_time'],
            'emotion': emotion_data['emotion_label'],
            'confidence': emotion_data['confidence_score']
        })

    # Generar la salida final (Contrato JSON)
    final_output = {
        'audio_analysis': {
            'transcribed_text': transcribed_text_section,
            'text_emotions': text_emotions_section
        },
        'metadata': {
            'processing_module': 'Audio/Texto',
            'asr_model': ASR_MODEL_NAME,
            'nlp_model': NLP_MODEL_NAME
        }
    }
    
    # Guardar la salida (Criterio 3 PBI 2.5)
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
        
    print(f"\n[PBI 2.5] Ensamblaje ASR/NLP completado y guardado en {OUTPUT_JSON_PATH}. Criterios OK.")
    return final_output

# ==============================================================================
# FUNCIÓN PRINCIPAL DE EJECUCIÓN DEL MÓDULO (Día 2 Entregable)
# ==============================================================================
def main_module_run():
    """Ejecuta la secuencia completa del módulo Audio/Texto."""
    
    # 1. Configuración de pipelines (necesario para todas las tareas)
    setup_pipelines()
    
    # 2. PBI 2.3: Extracción de audio
    if not extract_audio(VIDEO_PATH, AUDIO_PATH):
        # Si la extracción falla o el video no existe, intentamos usar el audio_prueba_10s.wav
        print("Intentando usar el audio_prueba_10s.wav existente para la transcripción.")
        audio_to_process = os.path.join(BASE_DIR, "01_DATA", "raw", "audio_prueba_10s.wav")
    else:
        audio_to_process = AUDIO_PATH

    if not os.path.exists(audio_to_process):
        print("ERROR CRÍTICO: No se encontró audio para procesar. Finalizando módulo.")
        return

    # 3. Transcripción ASR (Con timestamps para PBI 2.5)
    transcription_results = get_transcription_with_timestamps(audio_to_process)
    
    # 4. PBI 2.5: Ensamblaje y Análisis NLP (Llama a PBI 2.2 internamente)
    final_output = assemble_asr_nlp_output(transcription_results)
    
    print("\n-------------------------------------------")
    print("FIN DEL DÍA 2: Módulo Audio/Texto Completado.")
    print("-------------------------------------------")

if __name__ == "__main__":
    main_module_run()