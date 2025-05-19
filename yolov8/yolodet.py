from picamera2 import Picamera2
import cv2
import time
from ultralytics import YOLO

# Initialisation de la caméra
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(1)

# Charger le modèle YOLOv8
model = YOLO("yolov8n.pt")  # Téléchargement automatique si nécessaire

# Boucle de détection
try:
    while True:
        frame = picam2.capture_array()
        
        # Si image à 4 canaux, on enlève le dernier canal (alpha)
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]

        # Optionnel : convertir en RGB si nécessaire (selon format caméra)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # si BGRA -> BGR
        # ou
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)  # si RGBA -> RGB

        results = model.predict(source=frame, show=True, conf=0.5, classes=None)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Arrêt par l'utilisateur.")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
