import json
import os
import sys
import subprocess

# --- AJUSTE DE RUTAS PARA IMPORTACIONES ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))

try:
    from utils.logger import get_logger
    from utils.helpers import validate_input_file
    log = get_logger("QA_Integracion_3.5")
except ImportError:
    print("ERROR: No se pudieron cargar las utilidades.")
    sys.exit(1)

def run_full_pipeline(video_name):
    """Ejecuta el pipeline completo usando el intérprete del entorno virtual."""
    log.info(f"--- Iniciando Prueba de Integración para: {video_name} ---")
    
    pipeline_script = os.path.join(current_dir, "..", "main_pipeline.py")
    
    try:
        # sys.executable usa el Python de tu venv_sia activo automáticamente
        result = subprocess.run([sys.executable, pipeline_script], capture_output=True, text=True)
        
        if result.returncode == 0:
            log.info(f"Pipeline ejecutado con éxito para {video_name}")
            return True
        else:
            log.error(f"Error en la ejecución del pipeline: {result.stderr}")
            return False
    except Exception as e:
        log.error(f"Fallo crítico en subproceso: {e}")
        return False
    
def validate_json_output(video_name):
    """Valida que el JSON generado no tenga datos faltantes (Criterio TCI 3.5)."""
    # Ruta según tu estructura 05_OUTPUTS
    base_path = os.path.dirname(os.path.dirname(current_dir))
    json_path = os.path.join(base_path, "05_OUTPUTS", "json_reports", "final_integration", f"{video_name}_FINAL.json")
    
    if not os.path.exists(json_path):
        log.error(f"Archivo de salida no encontrado: {json_path}")
        return False

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validar campos críticos del contrato 1.3
        if "events" not in data or len(data["events"]) == 0:
            log.error("El JSON no contiene eventos procesados.")
            return False
        
        for i, event in enumerate(data["events"]):
            if not event.get("transcribed_text"):
                log.warning(f"Segmento {i}: Texto de transcripción vacío.")
            if event.get("emotion_facial_mode") == "unknown":
                log.warning(f"Segmento {i}: Emoción facial no detectada.")
        
        log.info(f"Validación de formato exitosa para {video_name}_FINAL.json")
        return True

    except Exception as e:
        log.error(f"Error al parsear el JSON: {e}")
        return False

if __name__ == "__main__":
    # Prueba con al menos 2 videos de validación (Criterio TCI 3.5)
    videos_test = ["video_03", "video_04"]
    
    for vid in videos_test:
        # 1. Correr el proceso completo
        success_run = run_full_pipeline(vid)
        
        # 2. Validar el resultado
        if success_run:
            validate_json_output(vid)
        
    log.info("=== Fin de Pruebas de Integración TCI 3.5 ===")