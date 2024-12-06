
import cv2
import mediapipe as mp

# Paths to the models
AGE_PROTO = "models/age_deploy.prototxt"
AGE_MODEL = "models/age_net.caffemodel"
GENDER_PROTO = "models/gender_deploy.prototxt"
GENDER_MODEL = "models/gender_net.caffemodel"

# Load age and gender models
age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

# Check if models loaded successfully
if age_net.empty() or gender_net.empty():
    print("Failed to load age or gender model.")
else:
    print("Age and gender models loaded successfully!")

# Age and gender labels
AGE_LABELS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LABELS = ['Male', 'Female']

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)

# Initialize Camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform face detection
    results = face_detection.process(rgb_frame)

    if results.detections:  # If faces are detected
        for detection in results.detections:
            # Get bounding box of the face
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape

            # Original bounding box
            x_center = int((bboxC.xmin + bboxC.width / 2) * iw)  # X center of the bounding box
            y_center = int((bboxC.ymin + bboxC.height / 2) * ih)  # Y center of the bounding box

            # Calculate padding based on the size of the bounding box
            x_padding = int(bboxC.width * iw * 0.1)  # 10% padding relative to the face width
            yt_padding = int(bboxC.height * ih * 0.5)  # 50% padding relative to the face height (top) (to include forehead of the person)
            yb_padding = int(bboxC.height * ih * 0.05)  # 5% padding relative to the face height (bottom)

            # Calculate new bounding box with padding
            x = max(0, x_center - int((bboxC.width * iw) / 2) - x_padding)
            w = min(iw - x, int(bboxC.width * iw) + 2 * x_padding)
            y = max(0, y_center - int((bboxC.height * ih) / 2) - yt_padding)
            h = min(ih - y, int(bboxC.height * ih) + yt_padding + yb_padding)


            # Extract face region
            face = frame[y:y+h, x:x+w]

            # Ensure face dimensions are valid
            if face.size > 0:
                # Prepare input blob for DNN models
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

                # Predict gender
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = GENDER_LABELS[gender_preds[0].argmax()]

                # Predict age
                age_net.setInput(blob)
                age_preds = age_net.forward()
                age = AGE_LABELS[age_preds[0].argmax()]

                # Display results
                label = f"{gender}, {age}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show the frame regardless of detection results
    cv2.imshow('Prepoznavanje spola i dobi', frame)

    # Exit the loop on pressing ESC
    if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit
        break

cap.release()
cv2.destroyAllWindows()