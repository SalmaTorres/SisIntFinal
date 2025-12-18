import json
import os
import sys
import matplotlib.pyplot as plt

# --- AJUSTE DE RUTAS PARA IMPORTACIONES ---
# Agregamos la raíz de 02_CODE al path para encontrar 'utils'
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", ".."))

try:
    from utils.logger import get_logger
    from utils.helpers import create_output_directory, validate_input_file
    log = get_logger("Visualizador_PBI3.3")
except ImportError as e:
    print(f"ERROR: No se pudieron cargar las utilidades. {e}")
    sys.exit(1)

def generate_comparison_plot(json_final_path):
    """
    PBI 3.3: Genera un gráfico comparativo de emociones (Texto vs Facial) 
    basado en el tiempo (Eje X en segundos).
    """
    if not validate_input_file(json_final_path):
        return

    try:
        with open(json_final_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        events = data.get("events", [])
        if not events:
            log.warning("No hay eventos de integración para graficar.")
            return

        # 1. Preparación de datos (Eje X y series)
        timestamps = [e["start_time_sec"] for e in events]
        text_emotions = [e["emotion_text_nlp"] for e in events]
        face_emotions = [e["emotion_facial_mode"] for e in events]

        # 2. Configuración del Gráfico (Criterios PBI 3.3)
        plt.figure(figsize=(14, 7))
        
        # Líneas paralelas para comparación visual
        plt.step(timestamps, text_emotions, where='post', label='Emoción Texto', marker='o', linewidth=2)
        plt.step(timestamps, face_emotions, where='post', label='Emoción Facial', marker='x', linewidth=2, alpha=0.8)

        # 3. Formato y Etiquetas
        plt.title(f"Análisis Multimodal de Emociones: {os.path.basename(json_final_path)}", fontsize=14)
        plt.xlabel("Segundos del Video (Eje X)", fontsize=12)
        plt.ylabel("Categoría Emocional", fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.6)

        # 4. Guardar resultado en 04_OUTPUTS/visualizations/
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        output_dir = os.path.join(base_dir, "05_OUTPUTS", "visualizations")
        create_output_directory(output_dir)
        
        video_id = os.path.splitext(os.path.basename(json_final_path))[0]
        output_path = os.path.join(output_dir, f"{video_id}_comparison.png")
        
        plt.savefig(output_path)
        plt.close()
        
        log.info(f"PBI 3.3 COMPLETADO: Gráfico guardado en {output_path}")

    except Exception as e:
        log.error(f"Error en la generación visual: {e}")

if __name__ == "__main__":
    # Ajusta esta ruta a un archivo que ya tengas generado en el Día 3
    video_test = "video_03_FINAL.json"
    root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    path_ejemplo = os.path.join(root, "05_OUTPUTS", "json_reports", "final_integration", video_test)
    
    generate_comparison_plot(path_ejemplo)