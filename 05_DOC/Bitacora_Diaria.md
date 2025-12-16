# Bitácora de Desarrollo - Sprint 1 (Lunes)

**Líder Técnico:** Salma
**Documentador:** Vivi (responsable de terminar este registro)
**QA:** Carla

## 1. Configuraciones y Setup (PBI 1.1)
- **Decisión:** Se optó por Python 3.x y se creó el entorno virtual `venv_sia`.
- **Librerías Base:** Instalación de `tensorflow`, `deepface`, `opencv-python`, y `tf-keras` (debido al error de dependencia).
- **Verificación:** Se pasó el test `check_deepface.py`.

## 2. Diseño de Arquitectura (PBI 1.3 - Contrato de Interfaz)
- **Decisión:** Se definió la estructura de salida **JSON** como contrato de interfaz para la fusión multimodal.
- **Detalle:** La estructura se basa en una lista de objetos `events` dentro de un objeto principal, incluyendo métricas globales (`global_metrics`) y datos por segmento (`start_time_sec`, `emotion_facial_mode`, `emotion_text_nlp`).
- **Archivo de Contrato:** `04_OUTPUTS/output_structure_contract.json`

## 3. Desarrollos del Módulo ASR/Texto (PBI 1.2)
- **Librerías Instaladas:** Se instalaron las dependencias para la transcripción y el NLP: `torch`, `torchaudio`, `transformers`, `stable-ts`, y `moviepy`.
- **Verificación:** Se pasó el test `check_asr.py`, confirmando que el modelo ASR (Whisper) se descarga y transcribió el audio de prueba.
- **Riesgo Mitigado (Dependencia Crítica):** Se identificó y resolvió un error de `moviepy` debido a la falta de la herramienta de sistema **FFmpeg**. Se logró la instalación de FFmpeg y se actualizó la variable **PATH** del sistema operativo, permitiendo la correcta extracción de audio de los archivos `.mp4`.
- **Impacto en el Contrato:** El script de prueba ya incluye la lógica para extraer audio, garantizando que el campo `transcribed_text` se llenará correctamente según el Contrato JSON (PBI 1.3).

## 4. Desarrollos del Módulo CNN/Facial (PBI 2.1)

- **Responsable:** Salma (Líder)
- **Tarea Completada:** Se desarrolló el script `02_CODE/pbi2_1_cnn_extractor.py`.
- **Funcionalidad:** Este script utiliza OpenCV para la lectura de video y DeepFace para procesar cada frame.
- **Criterios de Confirmación Cumplidos:**
    * **Procesamiento de Video:** El script abre y procesa el `video_01.mp4` (TCI 1.4).
    * **Salida Intermedia:** Se genera el archivo `04_OUTPUTS/cnn_time_series.csv` con el timestamp y la emoción dominante por frame.
    * **Manejo de No-Rostro:** Se implementó la lógica `try-except` para excluir los frames donde DeepFace no detecta un rostro.
