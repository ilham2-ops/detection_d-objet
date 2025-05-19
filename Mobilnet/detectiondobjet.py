import cv2
import numpy as np
import time
from picamera2 import Picamera2

# Initialize the camera
def initialize_camera():
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(main={"size": (1280, 720)})
    picam2.configure(preview_config)
    picam2.start()
    time.sleep(1)  
    return picam2

# Load the pre-trained MobileNet SSD model
def load_model():
    modelFile = "MobileNetSSD_deploy.caffemodel"
    configFile = "MobileNetSSD_deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net

# Define the classes the model can detect
def get_classes():
    return {
        0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
        5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow',
        11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person',
        16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'
    }

# Process frame and detect objects
def detect_objects(frame, net, classes):
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
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def main():
    picam2 = initialize_camera()
    net = load_model()
    classes = get_classes()

    print("Object detection started. Press 'q' to quit.")
    
    try:
        while True:
            frame = picam2.capture_array()

            # Convert BGRA to BGR if needed
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            processed_frame = detect_objects(frame, net, classes)
            cv2.imshow("Object Detection", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Stopping the program...")
    finally:
        cv2.destroyAllWindows()
        picam2.stop()
        print("Program stopped")

if __name__ == "__main__":
    main()
