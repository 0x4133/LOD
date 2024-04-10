import cv2
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Function to extract features from an image
def extract_features(image):
    if image is None or image.size == 0:
        print("Error: Empty image passed to extract_features()")
        return None

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the image to a fixed size
    resized = cv2.resize(gray, (100, 100))
    # Flatten the image into a 1D array
    features = resized.flatten()
    return features

# Function to train the model
def train_model(data_dir):
    # Initialize lists to store features and labels
    features = []
    labels = []

    # Iterate over the annotated images and annotations
    for filename in os.listdir(data_dir):
        if filename.startswith("annotated_frame_") and filename.endswith(".jpg"):
            # Load the annotated image
            image_path = os.path.join(data_dir, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Error: Failed to load image {image_path}")
                continue

            # Load the corresponding annotations
            annotation_path = os.path.join(data_dir, "annotations_" + filename[16:-4] + ".json")
            with open(annotation_path, "r") as file:
                annotations = json.load(file)

            # Extract features and labels for each annotated object
            for annotation in annotations:
                rect = annotation["rectangle"]
                object_name = annotation["object_name"]

                # Extract the object region from the image
                x1, y1 = rect[0]
                x2, y2 = rect[1]

                # Check if the rectangle coordinates are valid
                if x1 >= 0 and y1 >= 0 and x2 < image.shape[1] and y2 < image.shape[0]:
                    object_image = image[y1:y2, x1:x2]

                    # Extract features from the object image
                    object_features = extract_features(object_image)

                    if object_features is not None:
                        # Append the features and label to the lists
                        features.append(object_features)
                        labels.append(object_name)
                    else:
                        print(f"Error: Failed to extract features from object {object_name} in {filename}")
                else:
                    print(f"Skipping invalid rectangle: {rect} in {filename}")

    # Convert features and labels to numpy arrays
    features = np.array(features)
    labels = np.array(labels)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Train the SVM model
    model = SVC(kernel="linear")
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")

    return model

# Function to detect objects in an image using the trained model
def detect_objects(model, image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform object detection using sliding window approach
    window_size = (100, 100)
    step_size = 20

    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            # Extract the window from the image
            window = image[y:y + window_size[1], x:x + window_size[0]]

            # Check if the window has the expected number of channels (3 for BGR)
            if window.shape[2] != 3:
                window = cv2.cvtColor(window, cv2.COLOR_GRAY2BGR)

            # Extract features from the window
            features = extract_features(window)

            # Predict the object label using the trained model
            label = model.predict([features])[0]

            # Draw a rectangle around the detected object
            cv2.rectangle(image, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image

# Directory containing the annotated images and annotations
data_dir = "/Users/aaron/PycharmProjects/cv2Bby/data"

# Train the model
model = train_model(data_dir)

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame from the webcam")
        break

    # Perform object detection on the frame
    result_frame = detect_objects(model, frame)

    # Display the result
    cv2.imshow("Live Object Detection", result_frame)

    # Check for 'q' key to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()