import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = load_model('my_model.h5')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for the model (resize, rescale, expand dimensions)
    frame_resized = cv2.resize(frame, (224, 224))
    frame_rescaled = frame_resized / 255.0
    frame_expanded = np.expand_dims(frame_rescaled, axis=0)

    # Predict
    predictions = model.predict(frame_expanded)

    # Use predictions to draw on the frame (customize this part as needed)
    # E.g., if prediction > some threshold, draw a rectangle or label

    cv2.imshow('Webcam View - Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
