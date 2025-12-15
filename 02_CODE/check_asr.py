import stable_whisper
import os

# Asegúrate de que el archivo de audio exista en esta ruta
AUDIO_PATH = "01_DATA/raw/test_audio.mp3" 

def check_whisper_transcription():
    """
    Verifica la instalación de Whisper/stable-ts probando la transcripción.
    """
    print("--- Verificación de ASR: Transcripción de Audio ---")
    
    if not os.path.exists(AUDIO_PATH):
        print(f"\n ERROR: No se encontró el archivo de audio de prueba en: {AUDIO_PATH}")
        print("INSTRUCCIÓN: Coloca un archivo de audio corto (.mp3, .wav) en '01_DATA/raw/' llamado 'test_audio.mp3'.")
        return

    try:
        # Cargamos el modelo base de Whisper (ligero y rápido para prueba)
        print("Cargando modelo Whisper (base)...")
        model = stable_whisper.load_model('base')

        # Transcribimos el audio
        print(f"Transcribiendo audio: {AUDIO_PATH}...")
        result = model.transcribe(AUDIO_PATH, verbose=False)
        
        transcription = result.text.strip()

        print("\n ¡ÉXITO! El modelo ASR funciona correctamente.")
        print(f"Transcripción obtenida: '{transcription}'")
        
        # Este es el dato clave que debes entregar a Salma en el Sprint 3
        print("\nRecordatorio: Este texto se mapeará al campo 'transcribed_text' del Contrato JSON.")
            
    except Exception as e:
        print(f"\n ERROR CRÍTICO al ejecutar Whisper: {e}")
        print("INSTRUCCIÓN: Verifica la instalación de Torch, Torchaudio y las librerías ASR/NLP.")

if __name__ == "__main__":
    check_whisper_transcription()