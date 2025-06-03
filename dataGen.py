from __future__ import annotations

import cv2
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# Initialize variables
freeze_frame: bool = False
start_point: Tuple[int, int] = (-1, -1)
end_point: Tuple[int, int] = (-1, -1)


@dataclass
class Annotation:
    rect: Tuple[Tuple[int, int], Tuple[int, int]]
    name: str


annotations: List[Annotation] = []

# Mouse callback function to handle clicks
def mouse_callback(event: int, x: int, y: int, flags: int, param) -> None:
    global freeze_frame, start_point, end_point

    if event == cv2.EVENT_LBUTTONDOWN:
        if not freeze_frame:
            freeze_frame = True
        elif start_point == (-1, -1):
            start_point = (x, y)
        else:
            end_point = (x, y)
            draw_rectangle()

def draw_rectangle() -> None:
    global start_point, end_point

    if start_point != (-1, -1) and end_point != (-1, -1):
        cv2.rectangle(frozen_img, start_point, end_point, (0, 255, 0), 2)
        cv2.imshow('Webcam', frozen_img)

        object_name = input("Enter the object name: ")
        annotations.append(Annotation((start_point, end_point), object_name))

        start_point = (-1, -1)
        end_point = (-1, -1)

def save_annotations():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frame_path = Path(f"annotated_frame_{timestamp}.jpg")
    annotations_path = Path(f"annotations_{timestamp}.json")

    # Save the current frame
    cv2.imwrite(str(frame_path), frozen_img)
    print(f"\033[92mSaved frame to {frame_path}\033[0m")  # Green color

    # Save annotations to a JSON file
    data = [
        {"rectangle": ann.rect, "object_name": ann.name}
        for ann in annotations
    ]
    with annotations_path.open("w") as file:
        json.dump(data, file)
    print(f"\033[92mSaved annotations to {annotations_path}\033[0m")  # Green color

# Initialize webcam
def main() -> None:
    global frozen_img
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("\033[91mError: Unable to open the webcam.\033[0m")
        return

    cv2.namedWindow('Webcam')
    cv2.setMouseCallback('Webcam', mouse_callback)

    while True:
        if not freeze_frame:
            ret, frame = cap.read()

            if not ret:
                print("\033[91mError: Failed to capture frame from the webcam.\033[0m")
                break

            frozen_img = frame.copy()

        for ann in annotations:
            cv2.rectangle(frozen_img, ann.rect[0], ann.rect[1], (0, 255, 0), 2)
            cv2.putText(
                frozen_img,
                ann.name,
                (ann.rect[0][0], ann.rect[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

        cv2.imshow('Webcam', frozen_img)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            save_annotations()
            annotations.clear()
            freeze_frame = False
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
