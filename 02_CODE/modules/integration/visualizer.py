import matplotlib.pyplot as plt
import os
from utils.logger import get_logger

log = get_logger("Visualizador")

def generate_comparison_plot(integrated_data, output_path):
    log.info(f"Generando visualización con detección de cambios (PBI 4.1)...")
    
    times = [e['start_time_sec'] for e in integrated_data]
    text_emos = [e['emotion_text_nlp'] for e in integrated_data]
    face_emos = [e['emotion_facial_mode'] for e in integrated_data]

    plt.figure(figsize=(14, 7))
    
    # Graficar líneas principales
    plt.step(times, text_emos, where='post', label='Texto', marker='o', linewidth=2)
    plt.step(times, face_emos, where='post', label='Facial', marker='x',  linewidth=2)

    # MARCAR PUNTOS DE CAMBIO (PBI 4.1)
    for event in integrated_data:
        # En visualizer.py, dentro del loop de is_change_point:
        if event['is_change_point']:
            plt.axvline(x=event['start_time_sec'], color='red', linestyle='--', alpha=0.4)
            # Usamos el temporal_insight para saber qué cambió
            plt.text(event['start_time_sec'], 0.2, event['temporal_insight'], 
                    color='red', rotation=90, fontsize=7, fontweight='bold')
            
    plt.title("Detección de Cambios Emocionales y Transiciones", fontsize=14)
    plt.xlabel("Segundos")
    plt.ylabel("Emoción")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()