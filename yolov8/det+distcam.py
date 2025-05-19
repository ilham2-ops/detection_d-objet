import time
import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

# === Paramètres pour l'estimation de distance ===
# Focale en pixels (fournie par l'utilisateur)
FOCALE_PIXELS = 460

# Hauteurs de référence des objets courants (en mètres)
HAUTEURS_OBJETS = {
    "person": 1.7,         # personne
    "bicycle": 1.0,        # vélo
    "car": 1.5,            # voiture
    "motorcycle": 1.2,     # moto
    "airplane": 3.0,       # avion
    "bus": 3.0,            # bus
    "train": 3.5,          # train
    "truck": 2.5,          # camion
    "boat": 1.5,           # bateau
    "bench": 0.5,          # banc
    "bird": 0.2,           # oiseau
    "cat": 0.35,           # chat
    "dog": 0.5,            # chien
    "horse": 1.6,          # cheval
    "sheep": 0.8,          # mouton
    "cow": 1.5,            # vache
    "elephant": 3.0,       # éléphant
    "bear": 2.0,           # ours
    "zebra": 1.5,          # zèbre
    "giraffe": 4.0,        # girafe
    "backpack": 0.5,       # sac à dos
    "umbrella": 1.0,       # parapluie
    "chair": 0.8,          # chaise
    "couch": 0.8,          # canapé
    "potted plant": 0.5,   # plante en pot
    "bed": 0.6,            # lit
    "dining table": 0.8,   # table à manger
    "toilet": 0.8,         # toilette
    "tv": 0.5,             # télévision
    "laptop": 0.3,         # ordinateur portable
    "mouse": 0.05,         # souris
    "keyboard": 0.05,      # clavier
    "cell phone": 0.15,    # téléphone portable
    "bottle": 0.25,        # bouteille
    "cup": 0.15,           # tasse
    "bowl": 0.1,           # bol
    "refrigerator": 1.8    # réfrigérateur
}

# Valeur par défaut pour les objets non listés
HAUTEUR_PAR_DEFAUT = 0.5

class DetecteurObjets:
    def __init__(self, model_path="yolov8n.pt", confiance=0.4):
        """
        Initialisation du détecteur d'objets
        """
        print("Initialisation du système...")
        
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

    def calculer_distance(self, hauteur_pixels, nom_classe):
        """
        Calcule la distance en utilisant la hauteur de l'objet
        
        Args:
            hauteur_pixels: Hauteur de l'objet en pixels
            nom_classe: Nom de la classe de l'objet
            
        Returns:
            distance en mètres
        """
        if hauteur_pixels <= 0:
            return None
            
        # Récupérer la hauteur de référence pour cette classe
        hauteur_reelle = HAUTEURS_OBJETS.get(nom_classe, HAUTEUR_PAR_DEFAUT)
        
        # Formule de la distance: (hauteur réelle * focale) / hauteur en pixels
        distance = (hauteur_reelle * FOCALE_PIXELS) / hauteur_pixels
        
        return distance

    def demarrer_detection(self):
        """
        Démarre la boucle principale de détection
        """
        try:
            while True:
                # Capture de l'image
                frame = self.picam2.capture_array()
                
                # Conversion pour OpenCV (RGB vers BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Détection avec YOLOv8
                resultats = self.model(frame, conf=self.confiance)
                
                # Traitement des résultats
                for resultat in resultats:
                    boxes = resultat.boxes
                    for box in boxes:
                        # Récupération des coordonnées
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        
                        # Récupération de la classe et score
                        cls_id = int(box.cls[0].item())
                        nom_classe = resultat.names[cls_id]
                        confiance = box.conf[0].item()
                        
                        # Calcul de la hauteur en pixels
                        hauteur_pixels = y2 - y1
                        
                        # Estimation de la distance
                        distance = self.calculer_distance(hauteur_pixels, nom_classe)
                        
                        # Couleurs différentes selon le type d'objet (plus facile à visualiser)
                        if nom_classe == "person":
                            couleur = (0, 255, 0)  # vert pour les personnes
                        elif nom_classe in ["car", "truck", "bus", "motorcycle", "bicycle"]:
                            couleur = (255, 0, 0)  # bleu pour les véhicules
                        else:
                            couleur = (0, 165, 255)  # orange pour les autres objets
                        
                        # Affichage du rectangle
                        cv2.rectangle(frame, (x1, y1), (x2, y2), couleur, 2)
                        
                        # Texte à afficher
                        if distance:
                            texte = f"{nom_classe}: {distance:.2f}m"
                        else:
                            texte = f"{nom_classe}"
                        
                        # Fond pour le texte (meilleure lisibilité)
                        taille_texte = cv2.getTextSize(texte, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(frame, (x1, y1 - 25), (x1 + taille_texte[0], y1), (0, 0, 0), -1)
                        
                        # Affichage du texte
                        cv2.putText(frame, texte, (x1, y1 - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Affichage d'une légende pour la distance
                cv2.putText(frame, "Focale: 460px", (10, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Affichage de l'image
                cv2.imshow("Detection d'objets avec distance", frame)
                
                # Sortie avec la touche 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("Arrêt par l'utilisateur...")
        finally:
            cv2.destroyAllWindows()
            self.picam2.stop()
            print("Programme terminé")

if __name__ == "__main__":
    detecteur = DetecteurObjets(model_path="yolov8n.pt", confiance=0.4)
    detecteur.demarrer_detection()
