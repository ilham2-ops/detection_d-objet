import numpy as np
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import time
import os

# Initialisation Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
time.sleep(2)

# Constantes
FOCALE_PIXELS = 450  # Calibrée sur un objet connu
REAL_OBJECT_WIDTH_CM = 4  # Largeur réelle moyenne (à ajuster selon l’objet)
MIN_BOX_WIDTH = 30       # Largeur min en pixels pour valider une détection
COOLDOWN_TIME = 30       # Nombre d'itérations avant nouvelle annonce

# Variables état
voice_enabled = True
last_announced = ""
cooldown = 0

# Fonction pour calculer la distance
def get_distance(pixel_width):
    if pixel_width == 0:
        return 0
    return (REAL_OBJECT_WIDTH_CM * FOCALE_PIXELS) / pixel_width

# Chargement modèle YOLOv8
model = YOLO("yolov8n.pt")

# Fenêtre affichage
cv2.namedWindow('Object Distance Measure', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Object Distance Measure', 960, 720)

# Zone centrale
center_x, center_y = 1280 // 2, 720 // 2
zone_size = 100

while True:
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

        if pixel_width < MIN_BOX_WIDTH:
            continue  # Ignorer petits objets

        # Vérifier si objet dans zone centrale
        if x1 < center_x < x2 and y1 < center_y < y2:
            detected_in_center = True

            distance = get_distance(pixel_width)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} | {distance:.1f} cm"
            cv2.putText(img, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Annonce vocale si activée et cooldown ok
            if voice_enabled and label != last_announced and cooldown == 0:
                print("Annonce :", label)
                os.system(f'espeak "{label}"')
                last_announced = label
                cooldown = COOLDOWN_TIME
        else:
            # Autres objets en bleu
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

    # Affichage état voix
    status = "Voix : Activée" if voice_enabled else "Voix : Désactivée"
    cv2.putText(img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)

    # Dessin croix et rectangle centre
    cv2.drawMarker(img, (center_x, center_y), (0, 0, 255),
                   markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    cv2.rectangle(img,
                  (center_x - zone_size, center_y - zone_size),
                  (center_x + zone_size, center_y + zone_size),
                  (0, 0, 255), 1)

    # Message si aucun objet au centre
    if not detected_in_center:
        cv2.putText(img, "Aucun objet centre detecte",
                    (center_x - 180, center_y - zone_size - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

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
