import cv2
import numpy as np
import os
import requests

# Function to download the necessary files if they don't exist
def download_yolo_files(yolo_dir):
    weights_url = "https://pjreddie.com/media/files/yolov3.weights"
    config_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
    names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    
    if not os.path.exists(yolo_dir):
        os.makedirs(yolo_dir)

    for filename, url in [("yolov3.weights", weights_url),
                          ("yolov3.cfg", config_url),
                          ("coco.names", names_url)]:
        filepath = os.path.join(yolo_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded {filename} successfully!")
            else:
                print(f"Failed to download {filename}.")

def load_yolo_model(yolo_dir):
    weights_path = os.path.join(yolo_dir, "yolov3.weights")
    config_path = os.path.join(yolo_dir, "yolov3.cfg")
    names_path = os.path.join(yolo_dir, "coco.names")
    
    net = cv2.dnn.readNet(weights_path, config_path)
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    output_layers = net.getUnconnectedOutLayersNames()
    return net, classes, output_layers
    
def detect_faces(img, confidence_threshold=0.5, nms_threshold=0.4):
    yolo_dir = os.path.join("models")
    download_yolo_files(yolo_dir)
    net, classes, output_layers = load_yolo_model(yolo_dir)

    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold and classes[class_id] == 'person':
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    faces_info = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            faces_info.append((x,y,w,h))

    return faces_info
