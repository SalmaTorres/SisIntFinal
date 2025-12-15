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
