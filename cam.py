import argparse
import cv2
from tensorflow.keras.models import load_model
import numpy as np

def main(model_path):
    model = load_model(model_path)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (224, 224))
        frame_rescaled = frame_resized / 255.0
        frame_expanded = np.expand_dims(frame_rescaled, axis=0)

        predictions = model.predict(frame_expanded)
        label = str(np.squeeze(predictions))
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Webcam View - Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run live classification")
    parser.add_argument("--model-path", default="my_model.h5", help="Path to Keras model")
    args = parser.parse_args()
    main(args.model_path)

