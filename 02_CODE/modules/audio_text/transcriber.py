import os
import torch
import ffmpeg
from transformers import pipeline
from utils.logger import get_logger
from utils.helpers import validate_input_file, create_output_directory

log = get_logger("Modulo_Transcripcion")

ASR_MODEL = "openai/whisper-small"
NLP_MODEL = "pysentimiento/robertuito-emotion-analysis"

ASR_PIPE = None
NLP_PIPE = None

def setup_pipelines(device_str):
    global ASR_PIPE, NLP_PIPE
    dev = 0 if device_str == "cuda" and torch.cuda.is_available() else -1
    
    if ASR_PIPE is None:
        log.info(f"Cargando Whisper en {device_str}...")
        ASR_PIPE = pipeline("automatic-speech-recognition", model=ASR_MODEL, device=dev)
    if NLP_PIPE is None:
        log.info(f"Cargando RoBERTuito en {device_str}...")
        NLP_PIPE = pipeline("text-classification", model=NLP_MODEL, device=dev)

def extract_audio(video_path, audio_path):
    if os.path.exists(audio_path):
        log.info(f"Audio existente: {os.path.basename(audio_path)}. Saltando extracción.")
        return True

    if not validate_input_file(video_path): return False
    create_output_directory(os.path.dirname(audio_path))
    
    try:
        ffmpeg.input(video_path).output(audio_path, acodec='pcm_s16le', ar='16000').overwrite_output().run(capture_stdout=True, capture_stderr=True)
        log.info("Audio extraído con FFmpeg.")
        return True
    except ffmpeg.Error as e:
        log.error(f"FFmpeg falló: {e.stderr.decode() if e.stderr else 'Error desconocido'}")
        return False

def get_transcription_and_emotion(audio_path):
    log.info("Procesando audio (ASR + NLP)...")
    result = ASR_PIPE(audio_path, return_timestamps=True, generate_kwargs={"language": "spanish"})
    
    chunks = []
    for chunk in result.get('chunks', []):
        text = chunk['text'].strip()
        if text:
            raw_res = NLP_PIPE(text)[0]
            label = raw_res['label']
            
            # Regla: others es igual a neutral
            if label == "others":
                label = "neutral"

            if label == "joy":
                label = "happy"

            if label == "sadness":
                label = "sad"

            if label == "anger":
                label = "angry"
                
            chunks.append({
                'start_time': chunk['timestamp'][0], 
                'end_time': chunk['timestamp'][1],
                'text': text, 
                'emotion': label, 
                'confidence': raw_res['score']
            })
    return chunks