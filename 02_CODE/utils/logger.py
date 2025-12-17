import logging
import sys
import os

def get_logger(module_name):
    """
    Configura y devuelve un logger estandarizado.
    - Muestra logs en CONSOLA.
    - Guarda logs en ARCHIVO dentro de 05_OUTPUTS/logs.
    """
    # 1. Definir rutas relativas para llegar a 05_OUTPUTS/logs
    # Subimos desde 02_CODE/utils/logger.py (2 niveles) hasta SISINTFINAL
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    LOG_DIR = os.path.join(BASE_DIR, "05_OUTPUTS", "logs")
    
    # Asegurarnos de que la carpeta exista
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Ruta del archivo de log
    LOG_FILE = os.path.join(LOG_DIR, "system_execution.log")

    # 2. Configurar el Logger
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    # El formato exigido por la rúbrica
    formatter = logging.Formatter(
        '[%(asctime)s] - [%(levelname)s] - [%(name)s] - %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Evitar duplicar handlers (para que no salga el mensaje repetido)
    if not logger.handlers:
        # A) Handler de Consola (Terminal)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # B) Handler de Archivo (Guardar en disco)
        # mode='a' significa "append" (agrega al final, no borra lo anterior)
        try:
            file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"ADVERTENCIA: No se pudo crear el archivo de log en {LOG_FILE}: {e}")

    return logger