import cv2
import numpy as np
import time
import os
from picamera2 import Picamera2

# Hauteurs réelles en cm pour estimation de distance
HAUTEURS_REELLES = {
    'person': 170, 'bicycle': 100, 'car': 150, 'motorbike': 120,
    'bus': 300, 'truck': 250, 'bottle': 25, 'chair': 80, 'dog': 50,
    'cat': 35, 'cow': 120, 'sheep': 80, 'bird': 20, 'horse': 160,
    'aeroplane': 300, 'train': 350, 'boat': 150, 'diningtable': 80,
    'pottedplant': 50, 'sofa': 80, 'tvmonitor': 60
}
HAUTEUR_PAR_DEFAUT = 100

# Ta focale réelle calibrée
FOCALE_PIXELS = 460
FACTEUR_CALIBRAGE = 1.0  # Ajustable si besoin

def initialize_camera():
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(preview_config)
    picam2.start()
    time.sleep(1)
    return picam2

def load_model():
    modelFile = "MobileNetSSD_deploy.caffemodel"
    configFile = "MobileNetSSD_deploy.prototxt"
    return cv2.dnn.readNetFromCaffe(configFile, modelFile)

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
    distance = (hauteur_reelle * FOCALE_PIXELS) / hauteur_pixels
    return distance * FACTEUR_CALIBRAGE

def detect_objects(frame, net, classes, last_announced, cooldown_counter, vocal_active):
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    height, width = frame.shape[:2]

    cv2.line(frame, (width//2-20, height//2), (width//2+20, height//2), (255,255,255), 2)
    cv2.line(frame, (width//2, height//2-20), (width//2, height//2+20), (255,255,255), 2)
    cv2.putText(frame, f"Focale: {FOCALE_PIXELS}px", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    if cooldown_counter > 0:
        cooldown_counter -= 1

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            class_name = classes.get(class_id, 'Unknown')

            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            hauteur_pixels = endY - startY
            distance = calculer_distance(hauteur_pixels, class_name)

            label = f"{class_name}: {confidence * 100:.1f}%"
            if distance:
                label += f" | {distance:.1f} cm"

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(frame, (startX, y - 20), (startX + text_size[0], y), (0, 0, 0), -1)
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            centre_x = (startX + endX) / 2
            centre_y = (startY + endY) / 2

            if (abs(centre_x - width/2) < width/5) and (abs(centre_y - height/2) < height/5):
                if distance:
                    texte_central = f"{class_name}: {distance:.1f} cm"
                    cv2.putText(frame, texte_central, (width//2 - 100, height - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                if vocal_active and cooldown_counter == 0 and class_name != last_announced:
                    print(f"Annonce : {class_name}")
                    os.system(f'espeak "{class_name}"')
                    last_announced = class_name
                    cooldown_counter = 30
                break

    return frame, last_announced, cooldown_counter

def main():
    picam2 = initialize_camera()
    net = load_model()
    classes = get_classes()
    last_announced = ""
    cooldown_counter = 0
    vocal_active = False

    print("Appuie sur 'q' pour quitter, 'v' pour activer/désactiver la voix.")

    try:
        while True:
            frame = picam2.capture_array()
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            frame, last_announced, cooldown_counter = detect_objects(
                frame, net, classes, last_announced, cooldown_counter, vocal_active)

            bouton_color = (0, 255, 0) if vocal_active else (0, 0, 255)
            cv2.rectangle(frame, (10, 40), (200, 80), bouton_color, -1)
            cv2.putText(frame, "Voix ON" if vocal_active else "Voix OFF", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Detection avec audio", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('v'):
                vocal_active = not vocal_active

    except KeyboardInterrupt:
        print("Arrêt utilisateur.")
    finally:
        cv2.destroyAllWindows()
        picam2.stop()
        print("Programme terminé.")

if __name__ == "__main__":
    main()
