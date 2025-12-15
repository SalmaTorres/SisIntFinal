# Sistema de Análisis de Entrevistas (Práctica Integrada Avanzada)

## 1. Setup del Entorno y Reproducibilidad

1.  **Clonar Repositorio:**
    ```bash
    git clone [https://aws.amazon.com/es/what-is/repo/](https://aws.amazon.com/es/what-is/repo/)
    cd Sistema_Analisis_Entrevistas
    ```
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