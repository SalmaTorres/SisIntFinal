import cv2
from deepface import DeepFace
import pandas as pd 

VIDEO_PATH = "01_DATA/raw/video_01.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
frame_data = [] # Lista para almacenar la serie temporal

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break # Fin del video

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    
    emotion_result = "no_face" # Default si no se detecta
    
    print(f"--- Procesando Frame ID: {frame_count} en {timestamp_sec:.2f} segundos ---") # <-- Agrega esto

    try:
        # Analiza el frame para emoción
        # La función devuelve una lista de diccionarios si detecta múltiples rostros
        analysis = DeepFace.analyze(
            frame, 
            actions=['emotion'], 
            enforce_detection=True # Forzar detección, si falla, lanza excepción
        )
        
        # Asumiendo un solo rostro dominante por frame
        emotion_result = analysis[0]['dominant_emotion']
        
        # Si se detecta un rostro, registrar el dato
        frame_data.append({
            'frame_id': frame_count,
            'timestamp_sec': timestamp_sec,
            'emotion': emotion_result
        })
        
    except Exception as e:
        # Si DeepFace falla (ej. no hay rostro o es muy pequeño), lo ignoramos
        # Esto cumple con "Se excluyen los frames donde no se detecta rostro."
        print(f"Frame {frame_count}: No se detectó rostro o error: {e}") 

cap.release()
df = pd.DataFrame(frame_data)
df.to_csv("04_OUTPUTS/cnn_time_series.csv", index=False)
print("Serie temporal facial guardada.")