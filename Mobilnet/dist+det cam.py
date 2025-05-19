import cv2
import numpy as np
import time
from picamera2 import Picamera2

# Hauteurs réelles des objets en cm pour l'estimation de distance
HAUTEURS_REELLES = {
    'person': 170,        # Personne adulte moyenne (170 cm)
    'bicycle': 100,       # Vélo
    'car': 150,           # Voiture
    'motorbike': 120,     # Moto
    'bus': 300,           # Bus
    'truck': 250,         # Camion (non dans la liste MobileNet SSD mais utile)
    'bottle': 25,         # Bouteille
    'chair': 80,          # Chaise
    'dog': 50,            # Chien moyen
    'cat': 35,            # Chat
    'cow': 140,           # Vache
    'sheep': 80,          # Mouton
    'bird': 20,           # Oiseau
    'horse': 160,         # Cheval
    'aeroplane': 300,     # Avion (sur le sol)
    'train': 350,         # Train
    'boat': 150,          # Bateau
    'diningtable': 80,    # Table à manger
    'pottedplant': 50,    # Plante en pot
    'sofa': 80,           # Canapé
    'tvmonitor': 40       # Moniteur TV
}

# Valeur par défaut pour les objets non listés
HAUTEUR_PAR_DEFAUT = 100  # cm

# Focale de la caméra (à calibrer)
FOCALE_PIXELS = 800  # Valeur approximative, à ajuster en fonction de votre caméra

# Caméra
def initialize_camera():
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(main={"size": (1280, 720)})
    picam2.configure(preview_config)
    picam2.start()
    time.sleep(1)
    return picam2

# Chargement du modèle
def load_model():
    modelFile = "MobileNetSSD_deploy.caffemodel"
    configFile = "MobileNetSSD_deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net

def get_classes():
    return {
        0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
        5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow',
        11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person',
        16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'
    }

# Calcul de la distance en utilisant la taille apparente
def calculer_distance(hauteur_pixels, classe):
    """
    Calcule la distance en utilisant la hauteur apparente de l'objet
    
    Args:
        hauteur_pixels: Hauteur de l'objet en pixels
        classe: Nom de la classe de l'objet
        
    Returns:
        distance en cm
    """
    if hauteur_pixels <= 0:
        return None
        
    # Récupérer la hauteur de référence pour cette classe
    hauteur_reelle = HAUTEURS_REELLES.get(classe, HAUTEUR_PAR_DEFAUT)
    
    # Formule de la distance: (hauteur réelle * focale) / hauteur en pixels
    distance = (hauteur_reelle * FOCALE_PIXELS) / hauteur_pixels
    
    return distance

# Détection avec distance
def detect_objects(frame, net, classes):
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    height, width = frame.shape[:2]
    
    # Dessiner une croix au centre de l'image
    cv2.line(frame, (width//2-20, height//2), (width//2+20, height//2), (0, 255, 255), 2)
    cv2.line(frame, (width//2, height//2-20), (width//2, height//2+20), (0, 255, 255), 2)
    
    # Afficher la focale utilisée
    cv2.putText(frame, f"Focale: {FOCALE_PIXELS}px", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            class_name = classes.get(class_id, 'Unknown')
            
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Calcul de la hauteur en pixels
            hauteur_pixels = endY - startY
            
            # Estimation de la distance
            distance = calculer_distance(hauteur_pixels, class_name)
            
            # Couleurs différentes selon le type d'objet
            if class_name == "person":
                couleur = (0, 255, 0)  # vert pour les personnes
            elif class_name in ["car", "truck", "bus", "motorbike", "bicycle"]:
                couleur = (255, 0, 0)  # bleu pour les véhicules
            else:
                couleur = (0, 165, 255)  # orange pour les autres objets
            
            # Préparation du texte
            label = f"{class_name}: {confidence * 100:.1f}%"
            if distance:
                label += f" | {distance:.1f} cm"
            
            # Affichage du rectangle
            cv2.rectangle(frame, (startX, startY), (endX, endY), couleur, 2)
            
            # Fond pour le texte (meilleure lisibilité)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            taille_texte = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (startX, y-20), (startX + taille_texte[0], y+5), (0, 0, 0), -1)
            
            # Affichage du texte
            cv2.putText(frame, label, (startX, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Si l'objet est proche du centre de l'image, afficher la distance en grand
            centre_obj_x = (startX + endX) / 2
            centre_obj_y = (startY + endY) / 2
            
            if (abs(centre_obj_x - width/2) < width/5) and (abs(centre_obj_y - height/2) < height/5):
                if distance:
                    texte_central = f"{class_name}: {distance:.1f} cm"
                    cv2.putText(frame, texte_central, (width//2 - 100, height - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    return frame

def main():
    picam2 = initialize_camera()
    net = load_model()
    classes = get_classes()
    
    print("Object detection with camera-based distance started. Press 'q' to quit.")
    print(f"Using focal length: {FOCALE_PIXELS} pixels (calibrate this value if needed)")
    
    try:
        while True:
            frame = picam2.capture_array()
            if frame.shape[2] == 4:  # Si format RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif frame.shape[2] == 3 and frame.dtype == np.uint8:  # Si format RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            processed_frame = detect_objects(frame, net, classes)
            
            cv2.imshow("Object Detection with Camera Distance", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()
        picam2.stop()
        print("Program stopped")

if __name__ == "__main__":
    main()
