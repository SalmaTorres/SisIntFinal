# Sistema de Análisis de Entrevistas (Práctica Integrada Avanzada)

## 1. Setup del Entorno y Reproducibilidad

1.  **Requisitos de Sistema:**
    * **FFmpeg:** Esta herramienta es obligatoria para el procesamiento de archivos de video (`.mp4`). Debe estar instalada en el sistema operativo y su ruta (`/bin`) debe estar añadida a la variable de entorno **PATH**.

2.  **Crear y Activar Entorno Virtual:**
    ```bash
    python -m venv venv_sia
    source venv_sia/bin/activate
    ```
3.  **Instalar Dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Verificación Rápida (Módulo CNN):**
    * Asegure la existencia de `01_DATA/raw/test_face.jpg`.
    * Ejecute: `python 02_CODE/check_deepface.py`

5.  **Verificación Rápida (Módulo ASR):**
    * Asegure la existencia de `01_DATA/raw/test_audio.mp3` o `test_video.mp4`.
    * Ejecute: `python 02_CODE/check_asr.py`