import cv2
import numpy as np
import time
from picamera2 import Picamera2
import RPi.GPIO as GPIO

# Initialiser le capteur ultrasonique
def setup_ultrasonic():
    GPIO.setmode(GPIO.BCM)
    GPIO_TRIGGER = 18
    GPIO_ECHO = 24
    GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
    GPIO.setup(GPIO_ECHO, GPIO.IN)
    return GPIO_TRIGGER, GPIO_ECHO

def measure_distance(GPIO_TRIGGER, GPIO_ECHO):
    GPIO.output(GPIO_TRIGGER, True)
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)

    StartTime = time.time()
    StopTime = time.time()

    while GPIO.input(GPIO_ECHO) == 0:
        StartTime = time.time()

    while GPIO.input(GPIO_ECHO) == 1:
        StopTime = time.time()

    TimeElapsed = StopTime - StartTime
    distance = (TimeElapsed * 34300) / 2
    return distance

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

# Détection avec distance
def detect_objects(frame, net, classes, GPIO_TRIGGER, GPIO_ECHO):
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    height, width = frame.shape[:2]

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")

            label = f"{classes.get(class_id, 'Unknown')}: {confidence * 100:.2f}%"

            # Mesurer la distance
            dist = measure_distance(GPIO_TRIGGER, GPIO_ECHO)
            label += f" | {dist:.1f} cm"

            # Affichage
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def main():
    picam2 = initialize_camera()
    net = load_model()
    classes = get_classes()
    GPIO_TRIGGER, GPIO_ECHO = setup_ultrasonic()

    print("Object detection with distance started. Press 'q' to quit.")

    try:
        while True:
            frame = picam2.capture_array()
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            processed_frame = detect_objects(frame, net, classes, GPIO_TRIGGER, GPIO_ECHO)
            cv2.imshow("Object Detection with Distance", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        GPIO.cleanup()
        cv2.destroyAllWindows()
        picam2.stop()
        print("Program stopped")

if __name__ == "__main__":
    main()
