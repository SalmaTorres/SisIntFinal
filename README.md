# Sistema de Análisis de Entrevistas (Práctica Integrada Avanzada)

Este proyecto implementa un sistema inteligente multimodal que analiza entrevistas combinando el reconocimiento facial de emociones (CNN), transcripción y análisis de texto (Transformers), y análisis temporal (Series de Tiempo), utilizando modelos preentrenados para la integración inteligente.

---

## 1. Setup del Entorno y Reproducibilidad

El sistema requiere las siguientes dependencias externas y de Python para su correcto funcionamiento.

### 1.1. Pre-requisito de Sistema (FFmpeg) 

El módulo de Transcripción (ASR/Whisper) depende de la herramienta de sistema **FFmpeg** para la decodificación de archivos de audio. Debe instalar FFmpeg y añadir la carpeta de los ejecutables (`bin`) a la **Variable de Entorno PATH** de su sistema operativo.

### 1.2. Configuración de Python

1.  **Clonar Repositorio:**
    ```bash
    git clone [https://github.com/SalmaTorres/SisIntFinal](https://github.com/SalmaTorres/SisIntFinal)
    cd SisIntFinal
    ```
2.  **Crear y Activar Entorno Virtual:**
    ```bash
    python -m venv venv_sia
    # Para Windows (PowerShell):
    .\venv_sia\Scripts\activate.ps1
    # Para Linux/Mac:
    source venv_sia/bin/activate
    ```
3.  **Instalar Dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

### 1.3. Verificación de Módulos (Día 1)

**Verificación Rápida (Módulo CNN/DeepFace):** 
    * Asegure la existencia de `01_DATA/raw/test_face.jpg`.
    * Ejecute: `python 02_CODE/check_deepface.py` (Debe detectar una cara).
**Verificación Rápida (Módulo ASR/Whisper):** 
    * Asegure la existencia de `01_DATA/raw/audio_prueba_10s.wav` (formato WAV PCM 16kHz).
    * Ejecute: `python 02_CODE/module_audio_text.py` (Debe devolver la transcripción).

---

## 2. Modelos Preentrenados y Estrategia

La solución se basa en la integración de modelos preentrenados para optimizar el tiempo de desarrollo (5 días).

| Componente | Modelo Preentrenado Sugerido | Propósito | Requisito Técnico |
| :--- | :--- | :--- | :--- |
| **Reconocimiento Facial** | DeepFace / FER-2013 pretrained  | Detección y extracción de emociones por frame. | Modelo CNN Preentrenado  |
| **Transcripción Audio (ASR)** | Whisper (OpenAI)  | Convertir audio de video a texto. | Modelo ASR Preentrenado |
| **Análisis de Texto (NLP)** | ROBERTa-emotion / BERT multilingual  | Extracción de emociones del texto transcrito. | Modelo NLP Preentrenado (Transformers)  |
| **Análisis Temporal** | Pandas/Series de tiempo manual | Detección de cambios emocionales y congruencia multimodal. | Modelo GRU/LSTM (Series temporales)  |

---

## 3. Estructura del Proyecto

La arquitectura del proyecto sigue una estructura modular y organizada:
```
SisIntFinal/
├── 01_DATA/ 
│   └── raw/              # Videos y audios de validación creados por el equipo
├── 02_CODE/
│   ├── main_pipeline.py  # Script principal de ejecución y orquestación
│   └── module_audio_text.py # Implementación de ASR y NLP
│   └── check_deepface.py # Script de verificación del módulo CNN
├── 03_MODEL/
│   └── (Archivos de modelos preentrenados si son necesarios)
├── 04_OUTPUTS/
│   └── output_structure_contract.json # Contrato JSON de la interfaz multimodal
├── 05_DOC/
│   ├── Bitacora_Diaria.md   # Registro de progreso diario (Entregable Día 1)
│   └── Informe_Tecnico.pdf # Informe final del proyecto
└── requirements.txt      # Listado de dependencias Python
```

## 4. Ejecución del Sistema

Para ejecutar el pipeline completo, use el script principal `main_pipeline.py` una vez que todos los módulos estén integrados (Día 3):

```bash
python 02_CODE/main_pipeline.py

## 4.1. Uso de Módulos Independientes (Outputs del Día 2)

Antes de ejecutar el pipeline principal, se debe generar la salida de cada módulo por separado para asegurar la sincronización. Ambos módulos deben procesar el *mismo video* de validación (e.g., `video_entrevista_3.mp4`).

**A. Módulo CNN/Emociones (PBI 2.1 y PBI 2.4):**

1.  **Extracción de Serie Temporal (PBI 2.1):** Procesa el video *frame* por *frame* con DeepFace. (Asumiendo que tu script de PBI 2.1 se llama `pbi_2_1_extraction.py`).
    ```bash
    python 02_CODE/pbi_2_1_extraction.py
    ```
    > **Genera:** `04_OUTPUTS/cnn_time_series.csv`
2.  **Consolidación por Segmento (PBI 2.4):** Resume la serie temporal en segmentos, utilizando los límites definidos por el ASR/NLP.
    ```bash
    python 02_CODE/module_cnn_emotions.py
    ```
    > **Genera:** `04_OUTPUTS/cnn_module_output.json`

**B. Módulo Audio/Texto (PBI 2.3, 2.2, 2.5):**

* **Procesamiento Completo:** Extrae audio, realiza transcripción ASR con *timestamps* y aplica análisis NLP para emociones.
    ```bash
    python 02_CODE/module_audio_text.py
    ```
    > **Genera:** `04_OUTPUTS/audio_text_module_output.json`

---
---
