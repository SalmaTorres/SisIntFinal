import pandas as pd
from collections import Counter

def synchronize_data(transcription_data, csv_path):
    print("-> Sincronizando datos Multimodales...")
    df_faces = pd.read_csv(csv_path)
    integrated = []

    for seg in transcription_data:
        # Filtrar frames del video en el tiempo del texto
        mask = (df_faces['timestamp_sec'] >= seg['start_time']) & (df_faces['timestamp_sec'] <= seg['end_time'])
        segment_faces = df_faces.loc[mask]
        
        face_mode = "neutral"
        if not segment_faces.empty:
            face_mode = Counter(segment_faces['emotion'].tolist()).most_common(1)[0][0]

        integrated.append({
            "start_time_sec": seg['start_time'],
            "end_time_sec": seg['end_time'],
            "text": seg['text'],
            "emotion_text_nlp": seg['emotion'],
            "emotion_facial_mode": face_mode,
            "congruence": 1.0 if seg['emotion'].lower().split()[0] in face_mode.lower() else 0.0
        })
    return integrated