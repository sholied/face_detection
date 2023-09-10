import cv2
import numpy as np
from keras.models import load_model

# Load trained model
model = load_model('face_recognition_model.h5')  # Replace with the path to your trained model

# Initialize the face detection model
face_net = cv2.dnn.readNetFromCaffe("computer_vision/CAFFE_DNN/deploy.prototxt.txt", "computer_vision/CAFFE_DNN/res10_300x300_ssd_iter_140000.caffemodel")

# Initialize the webcam (change 0 to the path of your video file if using a video file)
video_capture = cv2.VideoCapture(0)

# Dictionary to map numeric labels to person names
labels_dict = {}
with open("labels.txt", "r") as file:
    lines = file.readlines()
    for line in lines:
        parts = line.strip().split(",")
        label_id = int(parts[0])
        label_name = parts[1]
        labels_dict[label_id] = label_name

while True:
    # Read a frame from the video feed
    ret, frame = video_capture.read()

    # Get the height and width of the frame
    (h, w) = frame.shape[:2]

    # Preprocess the frame
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 177, 123))

    # Pass the frame through the face detection model
    face_net.setInput(blob)
    detections = face_net.forward()

    # Create a list to store recognized faces
    recognized_faces = []

    # Iterate over the detected faces
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out face detections with confidence above a threshold
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
                    # Ensure the bounding box coordinates are valid
            if startX < 0:
                startX = 0
            if startY < 0:
                startY = 0

            # Extract the face region of interest (ROI) from the frame
            face_roi = frame[startY:endY, startX:endX]

            # Check if the ROI is empty or too small
            if face_roi.size == 0 or face_roi.shape[0] < 10 or face_roi.shape[1] < 10:
                continue

            # Resize and preprocess the ROI
            face_roi = cv2.resize(face_roi, (224, 224))


            # Extract the face region of interest (ROI) from the frame
            face_roi = cv2.resize(frame[startY:endY, startX:endX], (224, 224))
            face_roi = face_roi/255
            face_roi = np.expand_dims(face_roi, axis=0)

            # Recognize the face using the trained model
            predicted_probabilities = model.predict(face_roi)
            predicted_label = np.argmax(predicted_probabilities)
            confidence_res = predicted_probabilities[0][predicted_label]

            # Get the name from the labels_dict
            name = labels_dict.get(predicted_label, "Unknown")

            # Draw a bounding box around the detected face
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Display the name and confidence (accuracy)
            text = f"Name: {name} | Confidence: {confidence_res:.2f}"
            cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Store the recognized face information
            recognized_faces.append((name, confidence_res))

    # Display the frame with bounding boxes and names
    cv2.imshow('Video', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
