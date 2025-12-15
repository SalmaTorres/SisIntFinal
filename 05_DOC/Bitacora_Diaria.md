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

**Decisión:** Se definió la estructura de salida **JSON** como contrato de interfaz para la fusión multimodal.

**Detalle:** La estructura se basa en una lista de objetos `events` dentro de un objeto principal, incluyendo métricas globales (`global_metrics`) y datos por segmento (`start_time_sec`, `emotion_facial_mode`, `emotion_text_nlp`).

**Archivo de Contrato:** `04_OUTPUTS/output_structure_contract.json`

### 2.1. Interfaz entre Módulos (Reunión Final Día 1)

Se definieron los inputs y outputs preliminares para la integración del Día 3:

| Módulo de Origen | Módulo de Destino | Dato Transferido | Formato |
| :--- | :--- | :--- | :--- |
| Módulo Utilidades | Módulo Facial / Módulo ASR | Archivo de Video / Audio | Video (`.mp4`), Audio (`.wav`, 16kHz, Mono) |
| **Módulo ASR** | **Módulo Texto (NLP)** | Transcripción Completa | Cadena de texto (`string`) |
| Módulo Facial (CNN) | Módulo de Integración | Serie Temporal de Emociones | Lista de tuplas `(timestamp_sec, emoción)` |
| Módulo Texto (NLP) | Módulo de Integración | Emociones Clave del Texto | Lista de tuplas `(timestamp_sec, emoción, score)` |