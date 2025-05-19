import time
import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from ultralytics import YOLO

# Configuration du capteur ultrason
GPIO_TRIGGER = 18
GPIO_ECHO = 24

class DetecteurObjetDistance:
    def __init__(self, model_path="yolov8n.pt", confiance=0.4):
        """
        Initialisation du détecteur d'objets et du capteur ultrason
        """
        print("Initialisation du système...")
        
        # Configuration du GPIO pour l'ultrason
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
        GPIO.setup(GPIO_ECHO, GPIO.IN)
        GPIO.output(GPIO_TRIGGER, False)
        time.sleep(0.5)  # Stabilisation du capteur
        
        # Initialisation de la caméra
        self.picam2 = Picamera2()
        self.config = self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        self.picam2.configure(self.config)
        self.picam2.start()
        time.sleep(2)  # Attente pour l'initialisation de la caméra
        
        # Chargement du modèle YOLOv8
        try:
            self.model = YOLO(model_path)
            print(f"Modèle {model_path} chargé avec succès")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            exit(1)
            
        self.confiance = confiance
        print("Système prêt! Démarrage de la détection...")

    def mesurer_distance_ultrason(self):
        """
        Mesure la distance avec le capteur ultrason HC-SR04
        
        Returns:
            distance en mètres (ou None en cas d'erreur)
        """
        try:
            # Envoyer une impulsion de 10µs sur le trigger
            GPIO.output(GPIO_TRIGGER, True)
            time.sleep(0.00001)  # 10µs
            GPIO.output(GPIO_TRIGGER, False)
            
            debut_impulsion = time.time()
            timeout = debut_impulsion + 0.1  # timeout après 100ms
            
            # Attendre que l'echo passe à HIGH
            while GPIO.input(GPIO_ECHO) == 0:
                debut_impulsion = time.time()
                if debut_impulsion > timeout:
                    return None
            
            # Attendre que l'echo passe à LOW
            fin_impulsion = time.time()
            timeout = fin_impulsion + 0.1  # timeout après 100ms
            
            while GPIO.input(GPIO_ECHO) == 1:
                fin_impulsion = time.time()
                if fin_impulsion > timeout:
                    return None
            
            # Calcul de la durée de l'impulsion
            duree = fin_impulsion - debut_impulsion
            
            # Calcul de la distance (vitesse du son = 343m/s)
            # distance = (durée × vitesse du son) / 2
            distance = (duree * 343.0) / 2.0
            
            return distance
            
        except Exception as e:
            print(f"Erreur de mesure ultrason: {e}")
            return None

    def demarrer_detection(self):
        """
        Démarre la boucle principale de détection
        """
        try:
            while True:
                # Mesure avec le capteur ultrason
                distance_ultrason = self.mesurer_distance_ultrason()
                
                # Capture de l'image
                frame = self.picam2.capture_array()
                
                # Conversion pour OpenCV (RGB vers BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Détection avec YOLOv8
                resultats = self.model(frame, conf=self.confiance)
                
                # Affichage de la distance mesurée par ultrason
                if distance_ultrason is not None:
                    texte_ultrason = f"Distance ultrason: {distance_ultrason:.2f}m"
                    cv2.putText(frame, texte_ultrason, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "Erreur mesure ultrason", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Traitement des résultats de la détection d'objets
                for resultat in resultats:
                    boxes = resultat.boxes
                    for box in boxes:
                        # Récupération des coordonnées
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        
                        # Récupération de la classe et score
                        cls_id = int(box.cls[0].item())
                        nom_classe = resultat.names[cls_id]
                        confiance = box.conf[0].item()
                        
                        # Couleurs différentes selon le type d'objet
                        if nom_classe == "person":
                            couleur = (0, 255, 0)  # vert pour les personnes
                        elif nom_classe in ["car", "truck", "bus", "motorcycle", "bicycle"]:
                            couleur = (255, 0, 0)  # bleu pour les véhicules
                        else:
                            couleur = (0, 165, 255)  # orange pour les autres objets
                        
                        # Affichage du rectangle
                        cv2.rectangle(frame, (x1, y1), (x2, y2), couleur, 2)
                        
                        # Texte à afficher avec le nom de l'objet
                        texte = f"{nom_classe}: {confiance:.2f}"
                        
                        # Fond pour le texte (meilleure lisibilité)
                        taille_texte = cv2.getTextSize(texte, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(frame, (x1, y1 - 25), (x1 + taille_texte[0], y1), (0, 0, 0), -1)
                        
                        # Affichage du texte
                        cv2.putText(frame, texte, (x1, y1 - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Affichage d'une zone de visée au centre (pour indiquer l'axe du capteur ultrason)
                h, w = frame.shape[:2]
                cv2.line(frame, (w//2, h//2-20), (w//2, h//2+20), (0, 255, 255), 2)
                cv2.line(frame, (w//2-20, h//2), (w//2+20, h//2), (0, 255, 255), 2)
                
                # Affichage de l'image
                cv2.imshow("Detection d'objets et mesure ultrason", frame)
                
                # Sortie avec la touche 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("Arrêt par l'utilisateur...")
        finally:
            cv2.destroyAllWindows()
            self.picam2.stop()
            GPIO.cleanup()
            print("Programme terminé")


if __name__ == "__main__":
    try:
        detecteur = DetecteurObjetDistance(model_path="yolov8n.pt", confiance=0.4)
        detecteur.demarrer_detection()
    except Exception as e:
        print(f"Erreur fatale: {e}")
        GPIO.cleanup()
