from deepface import DeepFace
import os

# Define la ruta donde DEBES colocar una foto de prueba.
# Asegúrate de crear la carpeta 01_DATA/raw/ y poner una imagen con tu rostro.
IMAGE_PATH = "01_DATA/raw/test_face.jpg" 

def check_deepface_emotion():
    """
    Verifica la instalación de DeepFace analizando una imagen estática 
    para detectar la emoción dominante.
    """
    print("--- Verificación de DeepFace: Detección de Emociones (PBI 1.1) ---")
    
    # 1. Verifica la existencia del archivo de prueba
    if not os.path.exists(IMAGE_PATH):
        print(f"\n ERROR: No se encontró la imagen de prueba en la ruta: {IMAGE_PATH}")
        print("INSTRUCCIÓN: Por favor, crea la carpeta '01_DATA/raw' y guarda una foto con un rostro claro llamada 'test_face.jpg' dentro.")
        return

    try:
        # 2. Ejecuta DeepFace.analyze con la acción 'emotion'
        print(f"Analizando imagen: {IMAGE_PATH}...")
        
        # DeepFace descarga automáticamente los modelos necesarios si no existen.
        results = DeepFace.analyze(
            img_path=IMAGE_PATH, 
            actions=['emotion'], 
            enforce_detection=True  # Asegura que si no detecta rostro, falle
        )

        # 3. Muestra el resultado
        if results:
            result = results[0] # Tomar el primer rostro detectado
            dominant_emotion = result['dominant_emotion']
            
            print("\n ¡ÉXITO! DeepFace funciona correctamente.")
            print(f"Número de rostros detectados: {len(results)}")
            print(f"Emoción dominante detectada: {dominant_emotion.upper()}")
            print(f"Puntaje de certeza: {result['emotion'][dominant_emotion]:.2f}%")
            
        else:
            print("\n ADVERTENCIA: DeepFace no detectó rostros en la imagen. Intenta con otra foto más clara o de frente.")
            
    except Exception as e:
        print(f"\n ERROR CRÍTICO al ejecutar DeepFace: {e}")
        print("INSTRUCCIÓN: Verifica la instalación de TensorFlow y las librerías base.")

if __name__ == "__main__":
    check_deepface_emotion()