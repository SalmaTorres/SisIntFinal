import torch
from transformers import pipeline
import os

# --- Configuración del Modelo ---
MODEL_NAME = "openai/whisper-small" 

# 1. Obtener la ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 

# 2. Construir la ruta completa del audio
AUDIO_FILENAME = "audio_prueba_10s.wav"
AUDIO_PATH = os.path.join(BASE_DIR, "01_DATA", "raw", AUDIO_FILENAME)

def setup_and_test_asr():
    """
    Configura y prueba el modelo ASR Whisper (Hugging Face).
    """
    print(f" CRITERIO 1: Verificando instalación de Transformers y PyTorch...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    try:
        # Cargar el pipeline de ASR con el modelo Whisper
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=MODEL_NAME,
            device=device 
        )
        print(f"Modelo ASR {MODEL_NAME} cargado con éxito en {device}.")
        
    except Exception as e:
        print(f" Error al cargar el modelo ASR: {e}")
        return

    # --- Prueba de Transcripción (CRITERIO 2 y 3) ---
    print("\n CRITERIO 2 & 3: Probando transcripción de audio...")
    
    if not os.path.exists(AUDIO_PATH):
        print(f" ERROR CRÍTICO: Archivo no encontrado en {AUDIO_PATH}")
        return

    audio_to_process = AUDIO_PATH
    print(f"Procesando archivo: {audio_to_process}")

    try:
        # Realizar la transcripción
        result = asr_pipeline(
            audio_to_process, 
            chunk_length_s=30, 
            stride_length_s=(4, 2),
            ignore_warning=True # Ocultamos la advertencia de chunking para la prueba
        )
        transcription = result['text']
        
        print("\n--- Resultado de la Transcripción ---")
        print(f"Transcripción Obtenida: \"{transcription}\"")
        print("-----------------------------------")
        
        # Validación de éxito
        if len(transcription.strip()) > 0:
            print(" ¡Éxito! Transcripción realizada, Criterios 2 y 3 cumplidos.")
        else:
            print(" Advertencia: Transcripción vacía. Revisa si tu audio tiene voz.")

    except Exception as e:
        print(f" Error durante la transcripción: {e}")

if __name__ == "__main__":
    setup_and_test_asr()