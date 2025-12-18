import pandas as pd
import json
import os
from utils.logger import get_logger

log = get_logger("Modulo_Validacion")

def run_manual_validation(json_output_path, manual_csv_path):
    log.info("--- INICIANDO VALIDACIÓN TCI4.6 (IA vs Etiquetas Manuales) ---")
    
    if not os.path.exists(json_output_path) or not os.path.exists(manual_csv_path):
        log.error("Faltan archivos para la validación.")
        return None

    # 1. Cargar datos
    with open(json_output_path, 'r', encoding='utf-8') as f:
        ai_data = json.load(f)
    
    df_manual = pd.read_csv(manual_csv_path)
    video_id = ai_data['interview_id'].replace("INT-", "").split("-")[0].lower()
    
    # Filtrar etiquetas manuales para este video específico
    df_manual_video = df_manual[df_manual['video_id'].str.lower() == video_id]
    
    matches = 0
    total_validated = 0
    validation_results = []

    # 2. Comparar segmento por segmento
    for ai_event in ai_data['events']:
        start = ai_event['start_time_sec']
        
        # Buscar el segmento manual más cercano en tiempo
        manual_row = df_manual_video[
            (df_manual_video['start_time_sec'] >= start - 0.5) & 
            (df_manual_video['start_time_sec'] <= start + 0.5)
        ]
        
        if not manual_row.empty:
            total_validated += 1
            m_row = manual_row.iloc[0]
            
            # Verificar si el score de la IA coincide con el manual
            # (Damos un margen de error de 0.1 en el score)
            is_correct = abs(ai_event['congruence_score'] - m_row['manual_congruence']) <= 0.1
            
            if is_correct:
                matches += 1
            
            validation_results.append({
                "time": start,
                "ai_score": ai_event['congruence_score'],
                "manual_score": m_row['manual_congruence'],
                "status": "VALID" if is_correct else "INVALID"
            })

    # 3. Calcular Métrica de Robustez
    accuracy = (matches / total_validated * 100) if total_validated > 0 else 0
    
    log.info(f"Validación Finalizada. Robustez del Modelo: {accuracy:.2f}%")
    
    return {
        "robustness_accuracy": accuracy,
        "segments_validated": total_validated,
        "detailed_comparison": validation_results
    }