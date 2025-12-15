# Contrato de Interfaz de Datos (Output JSON Final)

Este documento define la estructura de datos obligatoria para el archivo JSON de salida final (generado por el pipeline completo en el Sprint 3). Esta estructura actúa como el **contrato de interfaz** que garantiza la fusión correcta de la información del Módulo CNN/Facial y el Módulo ASR/NLP.

**Archivo Contrato:** `output_structure_contract.json`

## 1. Estructura Principal del Documento

El documento JSON es un objeto único que contiene metadatos y una lista de eventos temporales.

| Campo | Tipo | Propósito | Módulo Fuente Principal |
| :--- | :--- | :--- | :--- |
| `interview_id` | `string` | Identificador único del archivo de entrevista. | General |
| `video_path` | `string` | Ruta relativa al archivo MP4 procesado. | General |
| `global_metrics` | `object` | Contiene el resumen cuantitativo de toda la entrevista (ej. congruencia promedio). | Sprint 4 (Análisis) |
| `events` | `array` | La serie temporal principal. Cada elemento es un segmento de audio/video. | Sprint 3 (Integración) |

---

## 2. Detalle de la Serie Temporal (`events`)

Cada objeto dentro del array `events` representa un segmento de la entrevista (generalmente definido por un cambio de orador o una pausa significativa en la transcripción).

| Campo | Tipo | Propósito | Módulo Fuente |
| :--- | :--- | :--- | :--- |
| `start_time_sec` | `float` | Momento de inicio del segmento (en segundos). **CRÍTICO para la sincronización.** | Módulo ASR (Vivi) |
| `end_time_sec` | `float` | Momento de finalización del segmento (en segundos). | Módulo ASR (Vivi) |
| `transcribed_text` | `string` | La transcripción literal del audio en ese segmento. | PBI 1.2 (Vivi) |
| `emotion_facial_mode` | `string` | La emoción facial más frecuente/dominante detectada en este segmento. | PBI 2.1 (Carla) |
| `emotion_facial_history` | `array` | **Serie temporal cruda de emociones por frame** dentro del segmento. Utilizado para análisis temporal (Sprint 4). | PBI 2.1 (Carla) |
| `emotion_text_nlp` | `string` | La emoción inferida directamente del `transcribed_text` mediante el modelo Transformer. | PBI 2.2 (Vivi) |
| `congruence_score` | `float` | Puntuación (0.0 a 1.0) que indica el acuerdo entre la emoción facial y la de texto. | PBI 4.2 (Salma) |
| `temporal_insight` | `string` | Notas o *insights* generados automáticamente (ej. "Pico de ansiedad", "Transición emocional abrupta"). | PBI 4.3 (Carla) |

---

## 3. Ejemplo de Segmento JSON

```json
{
  "start_time_sec": 10.5,
  "end_time_sec": 14.2,
  "transcribed_text": "Estoy un poco nervioso por el examen, pero preparado.",
  "emotion_facial_mode": "neutral",
  "emotion_facial_history": ["neutral", "neutral", "sadness", "neutral"],
  "emotion_text_nlp": "anxiety", 
  "congruence_score": 0.35,
  "temporal_insight": "Incongruencia: El texto expresa ansiedad, pero el rostro se mantuvo neutral."
}