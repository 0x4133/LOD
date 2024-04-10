import cv2
import json
from datetime import datetime

# Initialize variables
freeze_frame = False
start_point = (-1, -1)
end_point = (-1, -1)
rectangles = []
object_names = []

# Mouse callback function to handle clicks
def mouse_callback(event, x, y, flags, param):
    global freeze_frame, start_point, end_point

    if event == cv2.EVENT_LBUTTONDOWN:
        if not freeze_frame:
            freeze_frame = True
        elif start_point == (-1, -1):
            start_point = (x, y)
        else:
            end_point = (x, y)
            draw_rectangle()

def draw_rectangle():
    global start_point, end_point

    if start_point != (-1, -1) and end_point != (-1, -1):
        cv2.rectangle(frozen_img, start_point, end_point, (0, 255, 0), 2)
        cv2.imshow('Webcam', frozen_img)

        # Prompt the user to enter the object name
        object_name = input("Enter the object name: ")
        rectangles.append((start_point, end_point))
        object_names.append(object_name)

        start_point = (-1, -1)
        end_point = (-1, -1)

def save_annotations():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frame_path = f"annotated_frame_{timestamp}.jpg"
    annotations_path = f"annotations_{timestamp}.json"

    # Save the current frame
    cv2.imwrite(frame_path, frozen_img)
    print(f"\033[92mSaved frame to {frame_path}\033[0m")  # Green color

    # Save annotations to a JSON file
    annotations = [
        {"rectangle": rect, "object_name": name}
        for rect, name in zip(rectangles, object_names)
    ]
    with open(annotations_path, "w") as file:
        json.dump(annotations, file)
    print(f"\033[92mSaved annotations to {annotations_path}\033[0m")  # Green color

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("\033[91mError: Unable to open the webcam.\033[0m")  # Red color
    exit()

cv2.namedWindow('Webcam')
cv2.setMouseCallback('Webcam', mouse_callback)

while True:
    if not freeze_frame:
        ret, frame = cap.read()

        if not ret:
            print("\033[91mError: Failed to capture frame from the webcam.\033[0m")  # Red color
            break

        frozen_img = frame.copy()

    # Draw existing rectangles and object names on the current frame
    for rect, name in zip(rectangles, object_names):
        cv2.rectangle(frozen_img, rect[0], rect[1], (0, 255, 0), 2)
        cv2.putText(frozen_img, name, (rect[0][0], rect[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the image
    cv2.imshow('Webcam', frozen_img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        save_annotations()
        rectangles = []  # Reset the rectangles list for new annotations
        object_names = []  # Reset the object names list for new annotations
        freeze_frame = False
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()