import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained Faster R-CNN model
model = tf.keras.models.load_model('faster_rcnn.h5')

# Load the video
cap = cv2.VideoCapture('video.mp4')

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Pre-process the frame for the model
    frame = cv2.resize(frame, (800, 800))
    frame = np.expand_dims(frame, axis=0)

    # Run the Faster R-CNN model on the frame
    detections = model.predict(frame)

    # Get the bounding boxes and class labels from the detections
    boxes = detections[:, :4]
    labels = detections[:, 4:]

    # Convert the bounding boxes from [ymin, xmin, ymax, xmax] to [xmin, ymin, xmax, ymax]
    boxes[:, [0, 2]] = boxes[:, [1, 3]]
    boxes[:, [1, 3]] = boxes[:, [2, 0]]

    # Loop over the detections and draw the bounding boxes on the frame
    for i in range(detections.shape[0]):
        xmin, ymin, xmax, ymax = boxes[i, :]
        label = labels[i, :]
        confidence = label[np.argmax(label)]

        if confidence > 0.5:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv2.putText(frame, f'{np.argmax(label)}: {confidence:.2f}', (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame with the bounding boxes
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()


"""

Advantages and Drawbacks of R-CNN, Fast R-CNN, Faster R-CNN:

R-CNN, Fast R-CNN, and Faster R-CNN are all popular object detection algorithms, but each has its own advantages and disadvantages.

R-CNN (Regions with Convolutional Neural Networks):

Advantages:

First deep learning-based object detection approach that demonstrated high accuracy.
Can detect objects of different shapes and sizes in an image.
Drawbacks:

Computationally expensive due to the need to run a separate CNN on each region proposal.
Slow, with a high latency due to the sequential execution of multiple steps (region proposal generation, feature extraction, and classification).
Fast R-CNN:

Advantages:

Significantly faster than R-CNN due to the use of shared convolutional features for all region proposals.
Can use a deep neural network for both feature extraction and classification, resulting in improved accuracy.
Drawbacks:

Still relatively slow compared to later object detection approaches like Faster R-CNN.
Uses a large memory footprint due to the requirement to store all region proposals in memory.
Faster R-CNN:

Advantages:

Much faster than both R-CNN and Fast R-CNN due to the use of a Region Proposal Network (RPN) to generate region proposals in a single forward pass.
High accuracy due to the use of deep neural networks for both feature extraction and classification.
Drawbacks:

More complex than Fast R-CNN, with a larger number of trainable parameters.
Can be challenging to train, especially for complex datasets with many object categories and variations.
In summary, Faster R-CNN is the best choice when both speed and accuracy are important, but it can be more difficult to train and has a larger memory footprint than Fast R-CNN. Fast R-CNN is a good alternative when computational resources are limited. R-CNN is less commonly used today due to its slow speed, but it remains an important milestone in the development of object detection algorithms.
"""