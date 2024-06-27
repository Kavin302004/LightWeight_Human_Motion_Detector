

# In[11]:


import cv2
import numpy as np
from matplotlib import pyplot as plt


# Load YOLO
net = cv2.dnn.readNet(r"yolov3-tiny.weights", r"yolov3-tiny1.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO names
with open(r"coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define the human class
human_class_id = classes.index("person")



# Initialize video capture
cap = cv2.VideoCapture(0)
# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video capture")
    exit()

# Read the initial background frame
ret, background = cap.read()

# Check if the frame was read successfully
if not ret:
    print("Error: Could not read frame from camera")
    exit()
background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
background_gray = cv2.GaussianBlur(background_gray, (21, 21), 0)

def detect_motion(frame, background_gray, threshold=25):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    diff = cv2.absdiff(background_gray, gray_frame)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours) > 0

def detect_human(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    human_boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            if class_id == human_class_id:
                confidence = scores[class_id]
                if confidence > 0.4:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    human_boxes.append((x, y, w, h))
    return human_boxes


# In[13]:


while True:
    # Read frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame from camera")
        break
    
    # Convert frame to grayscale for motion detection
    background_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Check for motion
    if detect_motion(frame, background_gray):
        # Detect humans
        if detect_human(frame):
            print("Human Motion Detected!")
            human_boxes = detect_human(frame)
        
        # Draw rectangles around humans
            for (x, y, w, h) in human_boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, 'Human', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    # Display the frame
    cv2.imshow('Frame', frame)
    
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


