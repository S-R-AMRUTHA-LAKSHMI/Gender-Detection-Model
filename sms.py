from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv
from twilio.rest import Client  # Import Twilio client

# Load model in .keras format
model = load_model('gender_detection.keras', compile=False)

# Open video file
video_path = 'us.mp4'  # Replace with your video file path
video = cv2.VideoCapture(video_path)

classes = ['man', 'woman']

# Twilio configuration
account_sid = 'your_account_sid'  # Replace with your Twilio Account SID
auth_token = 'your_auth_token'    # Replace with your Twilio Auth Token
twilio_number = 'your_twilio_number'  # Replace with your Twilio phone number
to_number = 'recipient_number'  # Replace with the recipient's phone number

client = Client(account_sid, auth_token)  # Initialize Twilio client

# Function to send SMS
def send_sms(message):
    client.messages.create(
        body=message,
        from_=twilio_number,
        to=to_number
    )

# Loop through frames
while video.isOpened():

    # Read frame from video
    status, frame = video.read()

    if not status:
        break  # End of video

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
        message = "Lone Woman Detected"
        cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2)
        send_sms(message)  # Send SMS notification

    elif num_women > 0 and num_men > 0 and num_women < num_men:
        message = "Women Surrounded by Men"
        cv2.putText(frame, message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2)
        send_sms(message)  # Send SMS notification

    # Display output
    cv2.imshow("gender detection", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()
