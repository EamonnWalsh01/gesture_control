import mediapipe as mp
import cv2
import numpy as np
import os
from pathlib import Path

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def get_label_from_path(file_path):
    # Extract the label from the parent directory name
    return Path(file_path).parent.name

def make_skele_image(img_path, output_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image: {img_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    blank_image = np.zeros((height, width, 3), np.uint8)

    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    blank_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(20, 180, 90), thickness=2, circle_radius=1)
                )
            cv2.imwrite(output_path, blank_image)
            return output_path
        else:
            print(f"No hand detected in {img_path}")
            return None

def organize_dataset(source_folder, destination_folder):
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, source_folder)
                class_name = get_label_from_path(input_path)
                
                output_path = os.path.join(destination_folder, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                skele_path = make_skele_image(input_path, output_path)
                if skele_path:
                    print(f"Processed and saved: {skele_path}")
                else:
                    print(f"Failed to process: {input_path}")

    print("Dataset organization complete.")

# Usage
source_folder = 'leapGestRecog'
destination_folder = 'organized_dataset_skele'
organize_dataset(source_folder, destination_folder)
