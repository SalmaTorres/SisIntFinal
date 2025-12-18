# Sistema de Análisis de Entrevistas (Práctica Integrada Avanzada)

Este proyecto implementa un sistema inteligente multimodal de nivel avanzado para el análisis de entrevistas. Combina Reconocimiento Facial de Emociones (CNN), Transcripción y NLP (Transformers) y una capa de Inteligencia Temporal (Simulación GRU/LSTM) para detectar incongruencias y transiciones emocionales.

---

## 1. Setup del Entorno y Reproducibilidad
### 1.1. Requisito de Sistema (FFmpeg)
El sistema utiliza FFmpeg para la extracción y normalización de audio.

1. Descargue los ejecutables de gyan.dev.
2. Añada la carpeta (`bin`) a las Variables de Entorno (PATH) de su sistema.
3. Verifique con: (`ffmpeg -version`).

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

## 2. Arquitectura de Modelos e Inteligencia

El sistema utiliza una estrategia de Fusión Multimodal Tardía con lógica recurrente:

| Componente | Modelo | Funcion |
| :--- | :--- | :--- |
| **ASR** | (`openai/whisper-small`) | Transcripción con timestamps y forzado de idioma (ES) |
| **NLP** | (`robertuito-emotion-analysis`)  | Clasificación de emociones en texto (BERT-based). |
| **CNN** | (`DeepFace (VGG-Face)`)  | Extracción de serie temporal de rostros por frame. |
| **Fusion** | Simulación GRU/LSTM | Lógica de Hidden States para estabilidad temporal. |

---

### 2.1. Lógica Avanzada

- **Detección de Cambios**: Utiliza una memoria de estado emocional (hidden_state) para identificar transiciones abruptas en rostro o texto, evitando falsos positivos por ruido.
- **Métrica de Congruencia**: Calcula un Score de Acuerdo (0.0 a 1.0) basado en grupos de valencia (ej: "Joy" y "Surprise" tienen acuerdo parcial, "Joy" vs "Anger" tienen acuerdo cero).

---

## 3. Estructura del Proyecto

```
SisIntFinal/
├── 01_DATA/ 
│   └── raw/                 # Videos y audios de validación creados por el equipo
│   └── audio_clean/         # Audios extraídos (.wav)
│   └── series_temporales/   # CSVs generados por DeepFace
├── 02_CODE/
│   ├── main_pipeline.py     # Script principal de ejecución y orquestación
│   ├── modules/
│   │   ├── audio_text/      # transcriber.py (Whisper + RoBERTuito)
│   │   ├── visual/          # face_extractor.py (DeepFace)
│   │   └── integration/     # synchronizer.py (GRU), visualizer.py, validator.py
│   └── utils/               # logger.py, helpers.py
├── 05_OUTPUTS/
│   ├── json_reports/        # Reportes finales de integración 
│   ├── visualizations/      # Dashboards de comparación (.png)
│   └── logs/                # Trazabilidad del sistema
├── 05_DOC/
│   └── Bitacora_Diaria.md   # Registro de progreso diario
└── requirements.txt      # Listado de dependencias Python
```

---

## 4. Ejecución y Optimización

### 4.1. Pipeline End-to-End

Para procesar un video completo, configure el nombre en main_pipeline.py y ejecute:

```bash
python 02_CODE/main_pipeline.py
```
## 4.2. "Skip Logic" (Eficiencia)

El sistema detecta automáticamente si un video ya fue procesado:
- Si existe el (`.wav`), se salta la extracción de audio.
- Si existe el (`.csv`), se salta el análisis de DeepFace (ahorro masivo de tiempo en pruebas).

---

## 5. Análisis Multimodal

El reporte final (`_FINAL.json`) sigue una estructura de contrato avanzada:
- (`global_metrics`): Promedio de congruencia de toda la entrevista.
- (`emotion_facial_history`): Trazabilidad total de cada frame detectado para auditoría.
- (`temporal_insight`): Análisis cualitativo automático (ej: "Transición Abrupta en Rostro").

---

## 6. Validación de Robustez

1. Edite (`01_DATA/validation_labels.csv`) con etiquetas manuales (Ground Truth).
2. El sistema ejecutará automáticamente (`validator.py`) al final del pipeline.
3. Se generará un **Accuracy de Robustez** comparando el Score de la IA vs. el Score Manual.

---

## 7. Visualización Avanzada

El sistema genera un gráfico de dos niveles para facilitar la interpretación médica/psicológica:
1. **Nivel Superior**: Series temporales de emoción (Texto vs Rostro) con marcas rojas en los puntos de cambio detectados por la lógica recurrente.
2. **Nivel Inferior**: Gráfico de barras de Congruencia, codificado por colores (Verde: Acuerdo total, Rojo: Contradicción emocional).

## 8. Flujo de Transformación de Datos

* **Paso 1 (Extracción):** El video se divide en audio (`.wav`) y frames procesados (`.csv`).
* **Paso 2 (Análisis):** Whisper genera texto con marcas de tiempo; DeepFace genera etiquetas emocionales por segundo.
* **Paso 3 (Fusión):** Se realiza el mapeo 1:N (una frase para muchos frames faciales) para obtener la concordancia emocional.
* **Paso 4 (Salida):** Se genera el JSON final integrado y el gráfico comparativo de validación.
---
