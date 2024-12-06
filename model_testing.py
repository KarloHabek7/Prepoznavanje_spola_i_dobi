import cv2

# Paths to the models
AGE_PROTO = "models/age_deploy.prototxt"
AGE_MODEL = "models/age_net.caffemodel"
GENDER_PROTO = "models/gender_deploy.prototxt"
GENDER_MODEL = "models/gender_net.caffemodel"

# Load the models
age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

# Check if models loaded successfully
if age_net.empty() or gender_net.empty():
    print("Failed to load age or gender model.")
else:
    print("Age and gender models loaded successfully!")