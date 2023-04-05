import cv2
import numpy as np
import keras
from keras.models import load_model

# Load pre-trained YOLO model
model = load_model('yolo_drone_detection.h5')

# Load class labels for YOLO model
classes = ['drone']

# Open video capture object
cap = cv2.VideoCapture(0)

while True:
    # Read frame from video capture object
    ret, frame = cap.read()

    # Preprocess frame for YOLO model
    frame = cv2.resize(frame, (416,416))
    frame = frame.astype(np.float32) / 255.
    frame = np.expand_dims(frame, axis=0)

    # Run frame through YOLO model
    y_pred = model.predict(frame)

    # Postprocess model output to extract bounding boxes
    boxes = []
    for i in range(y_pred.shape[1]):
        scores = y_pred[0, i, 4:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            x1 = int(y_pred[0, i, 0] * frame.shape[1])
            y1 = int(y_pred[0, i, 1] * frame.shape[0])
            x2 = int(y_pred[0, i, 2] * frame.shape[1])
            y2 = int(y_pred[0, i, 3] * frame.shape[0])
            boxes.append([x1, y1, x2, y2])

    # Draw bounding boxes on frame
    for box in boxes:
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)

    # Display frame
    cv2.imshow('Drone Detection', frame)

    # Break loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object
cap.release()
cv2.destroyAllWindows()


"""
Explanation:
In this code, the pre-trained YOLO model is loaded using the load_model function from the Keras library. 
The model is then used to detect drones in each frame of a video stream captured using the cv2.VideoCapture class from the OpenCV library. 
The model outputs are postprocessed to extract bounding boxes around the detected drones, and these boxes are then drawn on the original frame using the cv2.rectangle function from OpenCV. 
The resulting frames are displayed using the cv2.imshow function, and the loop is broken if the user presses the 'q' key.
"""

"""
Advantages and drawbacks of YOLO:

Advantages of YOLO:

Speed: YOLO is known for its real-time object detection capability, which is crucial in many applications such as video surveillance, autonomous vehicles, and robotics.

Simplicity: YOLO is relatively simple compared to other object detection algorithms. It has a single network architecture for end-to-end object detection, making it easier to train and deploy.

Single shot detection: Unlike other object detection algorithms that require multiple proposals or stages, YOLO performs object detection in a single shot. This makes it faster and more efficient.

Improved accuracy: The latest versions of YOLO (v3 and beyond) have improved accuracy compared to the earlier versions.

Drawbacks of YOLO:

Low accuracy: Although YOLO has improved accuracy compared to earlier versions, it still lags behind other object detection algorithms in terms of accuracy.

Large file size: The pre-trained YOLO models are relatively large in size, making them unsuitable for deployment on devices with limited memory.

Poor handling of small objects: YOLO has difficulty detecting small objects, which can be a disadvantage in some applications.

Lack of interpretability: The workings of YOLO's single shot detection mechanism are somewhat opaque, making it harder for researchers to understand how the algorithm works and why it makes certain decisions.

"""