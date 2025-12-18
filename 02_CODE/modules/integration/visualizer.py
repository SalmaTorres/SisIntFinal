import matplotlib.pyplot as plt
import os
from utils.logger import get_logger

log = get_logger("Visualizador")

def generate_comparison_plot(integrated_data, output_path):
    log.info(f"Generando visualización con etiquetas originales...")
    
    # 1. Preparar los datos
    times = [e['start_time_sec'] for e in integrated_data]
    
    # Etiquetas de texto (vienen procesadas del transcriber)
    text_emos = [e['emotion_text_nlp'] for e in integrated_data]
    
    # Etiquetas faciales originales de DeepFace
    face_emos = [e['emotion_facial_mode'] for e in integrated_data]

    # 3. Configurar el gráfico
    plt.figure(figsize=(12, 7))
    
    # Graficar con estilo "step"
    plt.step(times, text_emos, where='post', label='Texto (RoBERTuito)', 
             marker='o', linewidth=2, color='#1f77b4')
    
    plt.step(times, face_emos, where='post', label='Facial (DeepFace)', 
             marker='x', linewidth=2, alpha=0.7, color='#ff7f0e', linestyle='--')

    # 4. Estética
    plt.title("Comparativa de Emociones (Etiquetas Originales)", fontsize=14, fontweight='bold')
    plt.xlabel("Segundos del video", fontsize=12)
    plt.ylabel("Emoción Detectada", fontsize=12)
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()

    # Guardar usando helpers (asumido en el flujo principal)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    log.info(f"Gráfico guardado en: {output_path}")