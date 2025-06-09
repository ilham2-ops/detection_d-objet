import numpy as np
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import time
import os

# Initialiser Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
time.sleep(2)

# Constantes
focal = 450  # Calibrée avec un objet connu
real_object_width = 4  # en cm
min_box_width = 30     # largeur min. en pixels pour valider un objet
voice_enabled = True
last_announced = ""
cooldown = 0
cooldown_time = 30

# Fonction pour calculer la distance
def get_distance(pixel_width):
    return (real_object_width * focal) / pixel_width if pixel_width != 0 else 0

# Charger modèle YOLO
model = YOLO("yolov8n.pt")

# Fenêtre d'affichage
cv2.namedWindow('Object Distance Measure', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Object Distance Measure', 960, 720)

# Définir zone centrale
center_x, center_y = 1280 // 2, 720 // 2
zone_size = 100

while True:
    # Capture et conversion
    frame = picam2.capture_array()
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    results = model(img)[0]
    detected_in_center = False

    if cooldown > 0:
        cooldown -= 1

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        pixel_width = x2 - x1

        if pixel_width < min_box_width:
            continue  # ignorer petits objets (bruit)

        # Vérifie si l'objet est dans la zone centrale
        if x1 < center_x < x2 and y1 < center_y < y2:
            detected_in_center = True

            # Distance
            distance = get_distance(pixel_width)

            # Dessin boîte verte + infos
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} | {distance:.1f} cm"
            cv2.putText(img, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Annonce vocale si activée
            if voice_enabled and label != last_announced and cooldown == 0:
                print("Annonce :", label)
                os.system(f'espeak "{label}"')
                last_announced = label
                cooldown = cooldown_time
        else:
            # Dessin des autres objets (non centrés) en bleu
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

    # Afficher état de la voix
    status = "Voix : Activée" if voice_enabled else "Voix : Désactivée"
    cv2.putText(img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)

    # Dessin de la croix + rectangle central
    cv2.drawMarker(img, (center_x, center_y), (0, 0, 255),
                   markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    cv2.rectangle(img,
                  (center_x - zone_size, center_y - zone_size),
                  (center_x + zone_size, center_y + zone_size),
                  (0, 0, 255), 1)

    # Si rien dans la zone centrale
    if not detected_in_center:
        cv2.putText(img, "Aucun objet centre detecte",
                    (center_x - 180, center_y - zone_size - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Affichage
    cv2.imshow("Object Distance Measure", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('v'):
        voice_enabled = not voice_enabled
        print("Voix activée" if voice_enabled else "Voix désactivée")

# Nettoyage
cv2.destroyAllWindows()
picam2.stop()
