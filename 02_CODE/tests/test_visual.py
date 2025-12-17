import unittest
import sys
import os
import pandas as pd
import json

# --- CONFIGURACIÓN DE RUTAS ---
# Agregamos la ruta 02_CODE al sistema para poder importar los módulos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Importamos la función de lógica pura de emotion_cnn
# Nota: Asegúrate de que tu archivo emotion_cnn.py tenga esta función accesible
from modules.visual.emotion_cnn import consolidate_emotions_by_segment

class TestVisualModule(unittest.TestCase):

    def setUp(self):
        print("\n[TEST] Iniciando prueba de Módulo Visual...")
        # Datos Dummy para simular lo que extrae DeepFace
        self.dummy_timeseries = [
            {'frame': 1, 'timestamp_sec': 1.0, 'emotion': 'happy', 'confidence': 90.0},
            {'frame': 15, 'timestamp_sec': 2.0, 'emotion': 'happy', 'confidence': 95.0},
            {'frame': 30, 'timestamp_sec': 3.0, 'emotion': 'sad', 'confidence': 60.0},
            {'frame': 45, 'timestamp_sec': 4.0, 'emotion': 'happy', 'confidence': 80.0},
            {'frame': 60, 'timestamp_sec': 5.0, 'emotion': 'angry', 'confidence': 50.0},
        ]

    def test_emotion_voting_logic(self):
        """
        PBI 2.4: Valida que la lógica de 'Moda' (Votación) elija la emoción mayoritaria.
        En self.dummy_timeseries hay: 3 happy, 1 sad, 1 angry. -> Debe ganar HAPPY.
        """
        start_time = 0.0
        end_time = 6.0
        
        result = consolidate_emotions_by_segment(self.dummy_timeseries, start_time, end_time)
        
        # Verificaciones
        print(f"   Input: 3 Happy, 1 Sad, 1 Angry -> Output: {result['emotion_facial_mode']}")
        
        self.assertEqual(result['emotion_facial_mode'], 'happy', "La moda debería ser 'happy'")
        self.assertAlmostEqual(result['confidence_facial_mode'], 0.6, msg="La confianza debería ser 3/5 (0.6)")
        self.assertEqual(len(result['emotion_facial_history']), 5, "Debe guardar el historial de los 5 frames")

    def test_empty_segment_logic(self):
        """
        PBI 2.4: Valida qué pasa si no hay caras en un segmento (ej: el sujeto se volteó).
        Debería retornar 'neutral' por defecto.
        """
        # Segmento de tiempo donde NO hay datos en dummy_timeseries (ej: seg 10 a 20)
        result = consolidate_emotions_by_segment(self.dummy_timeseries, 10.0, 20.0)
        
        self.assertEqual(result['emotion_facial_mode'], 'neutral')
        self.assertEqual(result['confidence_facial_mode'], 0.0)

    def test_csv_structure_compliance(self):
        """
        PBI 2.1: Valida que el archivo CSV generado (si existe) tenga las columnas obligatorias.
        Esto simula una verificación del output de face_extractor.py
        """
        # Buscamos un CSV real generado en OUTPUTS
        output_dir = os.path.join(BASE_DIR, "..", "05_OUTPUTS", "series_temporales")
        
        # Si no existe carpeta o archivos, saltamos el test (no es fallo, es falta de ejecución previa)
        if not os.path.exists(output_dir) or not os.listdir(output_dir):
            print("   [SKIP] No se encontraron CSVs reales para validar estructura. Ejecuta face_extractor.py primero.")
            return

        # Tomamos el primer CSV que encontremos
        csv_file = os.path.join(output_dir, os.listdir(output_dir)[0])
        try:
            df = pd.read_csv(csv_file)
            required_columns = ['frame', 'timestamp_sec', 'emotion', 'confidence']
            
            # Verificamos que todas las columnas requeridas estén presentes
            missing = [col for col in required_columns if col not in df.columns]
            self.assertEqual(len(missing), 0, f"Faltan columnas en el CSV: {missing}")
            
            print(f"   [OK] Estructura CSV validada correctamente: {csv_file}")
            
        except Exception as e:
            self.fail(f"El archivo CSV está corrupto o ilegible: {e}")

if __name__ == '__main__':
    unittest.main()