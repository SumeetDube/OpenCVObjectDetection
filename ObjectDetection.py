import cv2
import numpy as np

input = 'one-by-one-person-detection.mp4'
# Initialize the video capture object
cap = cv2.VideoCapture(input)  # Use 0 for the default camera, or provide a video file path


# Load the pre-trained object detection model
net = cv2.dnn.readNet("model/yolov3.weights", "model/yolov3.cfg")

# Get the names of the classes from the model
with open("model/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define the target object to detect
target_object = "person"  # Replace with the desired object name

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Preprocess the frame for object detection
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_output = net.forward(net.getUnconnectedOutLayersNames())

        # Iterate over the detected objects
        boxes = []
        confidences = []
        class_ids = []
        for output in layer_output:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] == target_object:
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    width = int(detection[2] * frame.shape[1])
                    height = int(detection[3] * frame.shape[0])
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    boxes.append([left, top, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maximum suppression to remove overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw the bounding boxes and track the object
        for i in indices:
            box = boxes[i]
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{classes[class_ids[i]]}: {confidences[i]:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

        # Display the resulting frame
        cv2.imshow("Object Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()