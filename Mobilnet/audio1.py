import cv2
import numpy as np
import time
import os
from picamera2 import Picamera2

FOCALE_PIXELS = 460
HAUTEUR_PAR_DEFAUT = 100

# Hauteurs réelles connues
HAUTEURS_REELLES = {
    'person': 170, 'bicycle': 100, 'car': 150, 'motorbike': 120, 'bus': 300,
    'truck': 250, 'bottle': 25, 'chair': 80, 'dog': 50, 'cat': 35, 'cow': 120,
    'sheep': 80, 'bird': 20, 'horse': 160, 'aeroplane': 300, 'train': 350,
    'boat': 150, 'diningtable': 80, 'pottedplant': 50, 'sofa': 80, 'tvmonitor': 60
}

def initialize_camera():
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(main={"size": (1280, 720)})
    picam2.configure(preview_config)
    picam2.start()
    time.sleep(1)
    return picam2

def load_model():
    return cv2.dnn.readNetFromCaffe(
        "MobileNetSSD_deploy.prototxt",
        "MobileNetSSD_deploy.caffemodel"
    )

def get_classes():
    return {
        0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
        5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow',
        11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person',
        16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'
    }

def calculer_distance(hauteur_pixels, classe):
    if hauteur_pixels <= 0:
        return None
    hauteur_reelle = HAUTEURS_REELLES.get(classe, HAUTEUR_PAR_DEFAUT)
    return (hauteur_reelle * FOCALE_PIXELS) / hauteur_pixels

def detect_objects(frame, net, classes, last_announced, cooldown, vocal_active):
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    height, width = frame.shape[:2]
    objet_centre_detecte = False

    # Affichage visuel
    cv2.line(frame, (width//2 - 20, height//2), (width//2 + 20, height//2), (255, 255, 255), 2)
    cv2.line(frame, (width//2, height//2 - 20), (width//2, height//2 + 20), (255, 255, 255), 2)
    cv2.rectangle(frame, (width//2 - 80, height//2 - 80), (width//2 + 80, height//2 + 80), (100, 100, 255), 2)
    cv2.putText(frame, f"Focale: {FOCALE_PIXELS}px", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    if cooldown > 0:
        cooldown -= 1

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.5:
            continue

        class_id = int(detections[0, 0, i, 1])
        class_name = classes.get(class_id, 'Unknown')

        if class_name not in HAUTEURS_REELLES:
            continue

        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
        startX, startY, endX, endY = box.astype("int")
        hauteur_pixels = endY - startY
        if hauteur_pixels < 30:  # éviter les très petits objets peu fiables
            continue

        distance = calculer_distance(hauteur_pixels, class_name)
        label = f"{class_name}: {confidence*100:.1f}% | {distance:.1f} cm"

        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

        center_x = (startX + endX) / 2
        center_y = (startY + endY) / 2

        # Si l'objet est au centre
        if abs(center_x - width/2) < 80 and abs(center_y - height/2) < 80:
            objet_centre_detecte = True
            cv2.putText(frame, f"{class_name}: {distance:.1f} cm", (width//2 - 100, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if vocal_active and cooldown == 0 and class_name != last_announced:
                os.system(f'espeak "{class_name}"')
                last_announced = class_name
                cooldown = 30
            break

    if not objet_centre_detecte:
        cv2.putText(frame, "Aucun objet au centre", (width//2 - 150, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)

    return frame, last_announced, cooldown

def main():
    picam2 = initialize_camera()
    net = load_model()
    classes = get_classes()

    last_announced = ""
    cooldown = 0
    vocal_active = True

    print("Appuie sur 'q' pour quitter, 'v' pour activer/désactiver la voix")

    try:
        while True:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            frame, last_announced, cooldown = detect_objects(
                frame, net, classes, last_announced, cooldown, vocal_active)

            # Bouton Voix ON/OFF visuel
            color = (0, 255, 0) if vocal_active else (0, 0, 255)
            cv2.rectangle(frame, (10, 40), (180, 80), color, -1)
            cv2.putText(frame, "Voix ON" if vocal_active else "Voix OFF", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Detection avec audio", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('v'):
                vocal_active = not vocal_active

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print("Programme terminé.")

if __name__ == "__main__":
    main()
