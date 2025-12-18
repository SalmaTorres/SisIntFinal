import matplotlib.pyplot as plt
import os
from utils.logger import get_logger

log = get_logger("Visualizador_Avanzado")

def generate_comparison_plot(integrated_data, output_path):
    log.info(f"Generando Dashboard Multimodal mejorado...")
    
    # 1. Preparación de datos
    times = [e['start_time_sec'] for e in integrated_data]
    text_emos = [e['emotion_text_nlp'] for e in integrated_data]
    face_emos = [e['emotion_facial_mode'] for e in integrated_data]
    scores = [e['congruence_score'] for e in integrated_data]

    # Crear figura con dos subplots (70% para emociones, 30% para score)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1]})

    # --- SUBPLOT 1: EMOCIONES (TEXTO VS ROSTRO) ---
    ax1.step(times, text_emos, where='post', label='Emoción Texto', 
             marker='o', linewidth=2.5, color='#1f77b4', zorder=3)
    ax1.step(times, face_emos, where='post', label='Emoción Rostro', 
             marker='x', linewidth=2, color='#ff7f0e', linestyle='--', alpha=0.8, zorder=2)

    # Añadir las líneas de "Cambio Abrupto" de forma discreta
    for e in integrated_data:
        if e.get('is_change_point'):
            # Línea vertical muy suave
            ax1.axvline(x=e['start_time_sec'], color='red', linestyle=':', alpha=0.3)
            # Etiqueta en la parte superior, fuera del área de las curvas
            ax1.text(e['start_time_sec'], ax1.get_ylim()[1], ' Δ Cambio', 
                     color='red', fontsize=8, fontweight='bold', rotation=0, va='bottom')

    ax1.set_title("Análisis Emocional Multimodal (Simulación GRU)", fontsize=14, pad=20)
    ax1.set_ylabel("Categoría Emocional")
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # --- SUBPLOT 2: SCORE DE CONGRUENCIA (MÉTRICA 4.2) ---
    # Usamos barras para mostrar el acuerdo en cada intervalo
    colors = ['green' if s >= 0.7 else 'orange' if s >= 0.3 else 'red' for s in scores]
    ax2.bar(times, scores, width=2.0, align='edge', color=colors, alpha=0.6, label='Nivel de Acuerdo')
    
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Congruencia")
    ax2.set_xlabel("Segundos del Video")
    ax2.grid(True, axis='y', linestyle=':', alpha=0.5)
    
    # Añadir una línea de meta para el score global si quieres
    avg_score = sum(scores) / len(scores) if scores else 0
    ax2.axhline(avg_score, color='blue', linestyle='--', alpha=0.5, label=f'Promedio: {avg_score:.2f}')
    ax2.legend(loc='upper right', fontsize='small')

    # Ajustar diseño
    plt.tight_layout()
    
    # Crear carpeta si no existe y guardar
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info(f"Dashboard profesional guardado en: {output_path}")