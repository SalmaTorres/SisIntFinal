import json
import os
import sys

# Ajuste de rutas para utilidades
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", ".."))

try:
    from utils.logger import get_logger
    from utils.helpers import validate_input_file
    log = get_logger("Analista_Avanzado")
except ImportError:
    print("ERROR: No se pudieron cargar las utilidades.")
    sys.exit(1)

def generate_insights(event):
    """
    Genera frases narrativas basadas en la congruencia y cambios emocionales.
    Cumple con el Criterio 1 del PBI 4.3.
    """
    score = event.get("congruence_score", 0.0)
    facial = event.get("emotion_facial_mode", "neutral")
    texto = event.get("emotion_text_nlp", "neutral")
    
    insight = ""
    
    if score < 0.4:
        insight = f"Insight: La congruencia es baja ({score}) debido a que el rostro expresa '{facial}' pero el texto indica '{texto}'."
    elif score < 0.7:
        insight = f"Insight: Congruencia moderada. Se observa una transición hacia '{facial}' en el rostro."
    else:
        insight = f"Insight: Alta coherencia emocional detectada en este segmento."
        
    return insight

def create_preliminary_report(json_path):
    """
    Lee el JSON procesado y actualiza los campos de insights.
    Genera un reporte preliminar (Criterio 2 del PBI 4.3).
    """
    if not validate_input_file(json_path):
        return

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        log.info(f"Generando insights para: {os.path.basename(json_path)}")
        
        # Procesar cada evento para inyectar la narrativa
        for event in data.get("events", []):
            event["temporal_insight"] = generate_insights(event)
            
        # Guardar el JSON actualizado (Reporte Preliminar)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
        log.info("PBI 4.3: Reporte preliminar e insights generados con éxito.")
        
    except Exception as e:
        log.error(f"Error en PBI 4.3: {e}")

if __name__ == "__main__":
    # Prueba con un archivo del Día 3
    test_file = os.path.join("05_OUTPUTS", "json_reports", "video_03_FINAL.json")
    create_preliminary_report(test_file)