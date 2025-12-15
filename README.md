# 🤖 Sistema de Análisis de Entrevistas (Práctica Integrada Avanzada)

[cite_start]Este proyecto implementa un sistema inteligente multimodal que analiza entrevistas combinando el reconocimiento facial de emociones (CNN), transcripción y análisis de texto (Transformers), y análisis temporal (Series de Tiempo), utilizando modelos preentrenados para la integración inteligente[cite: 18, 20].

---

## 1. Setup del Entorno y Reproducibilidad

El sistema requiere las siguientes dependencias externas y de Python para su correcto funcionamiento.

### 1.1. Pre-requisito de Sistema (FFmpeg) ⚠️

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

* [cite_start]**Verificación Rápida (Módulo CNN/DeepFace):** [cite: 39]
    * Asegure la existencia de `01_DATA/raw/test_face.jpg`.
    * Ejecute: `python 02_CODE/check_deepface.py` (Debe detectar una cara).
* [cite_start]**Verificación Rápida (Módulo ASR/Whisper):** [cite: 39]
    * Asegure la existencia de `01_DATA/raw/audio_prueba_10s.wav` (formato WAV PCM 16kHz).
    * Ejecute: `python 02_CODE/module_audio_text.py` (Debe devolver la transcripción).

---

## 2. Modelos Preentrenados y Estrategia

[cite_start]La solución se basa en la integración de modelos preentrenados para optimizar el tiempo de desarrollo (5 días)[cite: 20, 26, 29].

| Componente | Modelo Preentrenado Sugerido | Propósito | Requisito Técnico |
| :--- | :--- | :--- | :--- |
| **Reconocimiento Facial** | [cite_start]DeepFace / FER-2013 pretrained [cite: 24] | Detección y extracción de emociones por frame. | [cite_start]Modelo CNN Preentrenado [cite: 88] |
| **Transcripción Audio (ASR)** | [cite_start]Whisper (OpenAI) [cite: 24] | Convertir audio de video a texto. | [cite_start]Modelo ASR Preentrenado [cite: 89] |
| **Análisis de Texto (NLP)** | [cite_start]ROBERTa-emotion / BERT multilingual [cite: 24] | Extracción de emociones del texto transcrito. | [cite_start]Modelo NLP Preentrenado (Transformers) [cite: 90] |
| **Análisis Temporal** | [cite_start]Pandas/Series de tiempo manual [cite: 24] | Detección de cambios emocionales y congruencia multimodal. | [cite_start]Modelo GRU/LSTM (Series temporales) [cite: 91] |

---

## 3. Estructura del Proyecto

[cite_start]La arquitectura del proyecto sigue una estructura modular y organizada[cite: 71]:
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
## 4. Ejecución del Sistema

Para ejecutar el pipeline completo, use el script principal `main_pipeline.py` una vez que todos los módulos estén integrados (Día 3):

```bash
python 02_CODE/main_pipeline.py