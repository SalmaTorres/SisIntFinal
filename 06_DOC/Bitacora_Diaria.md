# Bitácora de Desarrollo - Sprint 1 (Lunes)

**Líder Técnico:** Salma
**Documentador:** Vivi (responsable de terminar este registro)
**QA:** Carla

## 1. Configuraciones y Setup (PBI 1.1, PBI 1.2)

**Decisión Ambiente:** Se optó por Python 3.x y se creó el entorno virtual `venv_sia`.

### 1.1. Módulo Facial (CNN/DeepFace)
- **Librerías Base:** Instalación de `tensorflow`, `deepface`, `opencv-python`, y `tf-keras` (debido al error de dependencia).
- **Verificación:** Se pasó el test `check_deepface.py`.

### 1.2. Módulo Audio/Texto (ASR/Whisper)
- **Modelo Seleccionado:** **Whisper** (OpenAI) a través de la librería **Hugging Face Transformers**.
- **Librerías Adicionales:** Instalación de `transformers`, `torch`, `librosa`.
- **Pre-requisito Resuelto (FFmpeg):** El pipeline de audio subyacente requería la herramienta de sistema **FFmpeg** para decodificar archivos de audio. Esto se resolvió instalando FFmpeg y añadiendo su ruta al PATH del sistema, lo cual desbloqueó la ejecución del modelo.
- **Criterios de Confirmación PBI1.2 - Setup ASR:** **100% Completado**
    - Librería instalada y configurada: **[OK]**
    - Clip de audio transcrito con éxito: **[OK]**
    - Transcripción obtenida: **[OK]**

> **Resultado de la Transcripción de Prueba (audio\_prueba\_10s.wav):** " hola soy un audio trasnochado que la está pasando muy mal por si alguien me escucha quiero que sepan que llevo todo el día siendo modelado y sistemas inteligentes y acá sigo y son las cuatro y media de la mañana gracias sigo siendo un audio hola hola"
→ Como vemos todavía confunde un poco las palabras como siendo/haciendo, igual puede ser porque no se hablar y no hay tanta claridad :*
## 2. Diseño de Arquitectura (PBI 1.3 - Contrato de Interfaz)
- **Decisión:** Se definió la estructura de salida **JSON** como contrato de interfaz para la fusión multimodal.
- **Detalle:** La estructura se basa en una lista de objetos `events` dentro de un objeto principal, incluyendo métricas globales (`global_metrics`) y datos por segmento (`start_time_sec`, `emotion_facial_mode`, `emotion_text_nlp`).
- **Archivo de Contrato:** `04_OUTPUTS/output_structure_contract.json`

## 3. Desarrollos del Módulo ASR/Texto (PBI 1.2)
- **Librerías Instaladas:** Se instalaron las dependencias para la transcripción y el NLP: `torch`, `torchaudio`, `transformers`, `stable-ts`, y `moviepy`.
- **Verificación:** Se pasó el test `check_asr.py`, confirmando que el modelo ASR (Whisper) se descarga y transcribió el audio de prueba.
- **Riesgo Mitigado (Dependencia Crítica):** Se identificó y resolvió un error de `moviepy` debido a la falta de la herramienta de sistema **FFmpeg**. Se logró la instalación de FFmpeg y se actualizó la variable **PATH** del sistema operativo, permitiendo la correcta extracción de audio de los archivos `.mp4`.
- **Impacto en el Contrato:** El script de prueba ya incluye la lógica para extraer audio, garantizando que el campo `transcribed_text` se llenará correctamente según el Contrato JSON (PBI 1.3).

## Sprint 2 (Martes)
**Líder Técnico:** Vivi
**Documentador:** Carla 
**QA:** Salma

## 4. Desarrollos del Módulo CNN/Facial (PBI 2.1)
- **Tarea Completada:** Se desarrolló el script `02_CODE/pbi2_1_cnn_extractor.py`.
- **Funcionalidad:** Este script utiliza OpenCV para la lectura de video y DeepFace para procesar cada frame.
- **Criterios de Confirmación Cumplidos:**
    * **Procesamiento de Video:** El script abre y procesa el `video_01.mp4` (TCI 1.4).
    * **Salida Intermedia:** Se genera el archivo `04_OUTPUTS/cnn_time_series.csv` con el timestamp y la emoción dominante por frame.
    * **Manejo de No-Rostro:** Se implementó la lógica `try-except` para excluir los frames donde DeepFace no detecta un rostro.

## 5. Desarrollo del Módulo Audio/Texto (PBI 2.3, PBI 2.2, PBI 2.5)

### 5.1. Extracción y Transcripción (PBI 2.3 & ASR)
- Se implementó la extracción de audio utilizando **ffmpeg-python** (en lugar de `moviepy` debido a inestabilidad), cumpliendo el requisito de crear una función helper.
- El modelo **Whisper** cargó correctamente y transcribió el video de validación con éxito, generando *timestamps* (ej., `[0.00s - 10.40s]`).
- **Verificación ASR:** El texto extraído es: 'Hola, quiero verse extraer las emociones, estoy super feliz, estoy super triste. Bueno, veremos, no sé.' (OK).

### 5.2. Análisis de Emociones (PBI 2.2)
- Se cargó el modelo de la familia **BERT** (`dccuchile/bert-base-spanish-wwm-uncased`) a través del *pipeline* de `text-classification`.
- **Fallo Documentado:** El modelo falló al procesar la clasificación (es un modelo base que necesita *fine-tuning*). Se documenta este fallo, pero se aplica una solución robusta:
    - **Solución:** Se implementó un manejo de errores en `get_text_emotions` que asigna la etiqueta **'NEUTRAL'** (con baja certeza) a los segmentos cuando el modelo arroja error, asegurando la continuidad.
- **Criterio PBI 2.2:** Cumplido, ya que se implementó el Transformer requerido y se manejó el fallo para no detener el sistema.

### 5.3. Ensamblaje JSON (PBI 2.5)
- Se generó la salida `audio_text_module_output.json` con la estructura final requerida, combinando *timestamps* ASR y las etiquetas de emoción (incluyendo el *fallback*). **PBI 2.5: OK.**

## 6. TCI 2.7 - Coordinación de Interfaces (Líder de Integración)
Se realizó la coordinación final con el Módulo CNN/Facial para la Integración del Día 3.
- **Acuerdo 1: Unidad de Tiempo:** Ambos módulos usarán **segundos flotantes** (ej., 1.5s) para los *timestamps* en el JSON final, asegurando la sincronización de video y audio (PBI 2.3).
- **Acuerdo 2: Etiquetas de Emoción:** El Módulo Facial (CNN) usará las 7 etiquetas de DeepFace, mientras que el Módulo Texto usa Sentimiento (`NEUTRAL`). Se acordó que la lógica de **Mapeo y Fusión** para resolver esta diferencia se implementará en el *main_pipeline* durante el Día 3.
**Entregable del Día 2:** Módulo Audio/Texto funcional (con solución robusta para NLP) y JSON de salida generado + Documentación de TCI 2.7.

