from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv

# Load model in .keras format
model = load_model('gender_detection.keras', compile=False)

# Open webcam
webcam = cv2.VideoCapture(0)

classes = ['man', 'woman']

# Loop through frames
while webcam.isOpened():

    # Read frame from webcam 
    status, frame = webcam.read()

    # Apply face detection
    faces, confidence = cv.detect_face(frame)

    # Initialize counters
    num_men = 0
    num_women = 0

    # Loop through detected faces
    for face in faces:

        # Get corner points of face rectangle        
        (startX, startY) = face[0], face[1]
        (endX, endY) = face[2], face[3]

        # Draw rectangle over face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Crop the detected face region
        face_crop = np.copy(frame[startY:endY, startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # Preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Apply gender detection on face
        conf = model.predict(face_crop)[0]

        # Get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]

        # Update counters
        if label == 'man':
            num_men += 1
        elif label == 'woman':
            num_women += 1

        # Prepare label and confidence for display
        label = "{}: {:.2f}%".format(label, conf[idx] * 100)
        Y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    # Check conditions and display results
    if num_women == 1 and num_men == 0:
        cv2.putText(frame, "Lone Woman Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2)
    elif num_women > 0 and num_men > 0 and num_women < num_men:
        cv2.putText(frame, "Women Surrounded by Men", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2)

    # Display output
    cv2.imshow("gender detection", frame)

    # Press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
