import os
import shutil
from pathlib import Path
import random

def get_gesture_from_path(file_path):
    # Get the gesture from the parent folder name
    return os.path.basename(os.path.dirname(file_path))

def organize_dataset(source_folder, destination_folder, test_split=0.2):
    image_files = []
    gestures = set()

    # Create destination folders
    for split in ['train', 'test']:
        os.makedirs(os.path.join(destination_folder, split), exist_ok=True)

    # Walk through all subdirectories
    for root, _, files in os.walk(source_folder):
        gesture = os.path.basename(root)
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(root, file)
                image_files.append((full_path, gesture))
                gestures.add(gesture)

    # Create a mapping of gestures to integers
    gesture_to_idx = {gesture: idx for idx, gesture in enumerate(sorted(gestures))}

    # Shuffle the image files
    random.shuffle(image_files)

    # Split into train and test sets
    split_index = int(len(image_files) * (1 - test_split))
    train_files = image_files[:split_index]
    test_files = image_files[split_index:]

    # Copy files to destination folders
    for split, file_list in [('train', train_files), ('test', test_files)]:
        for file_path, gesture in file_list:
            # Create gesture folder if it doesn't exist
            gesture_folder = os.path.join(destination_folder, split, gesture)
            os.makedirs(gesture_folder, exist_ok=True)

            # Keep the original filename
            original_name = os.path.basename(file_path)
            destination_path = os.path.join(gesture_folder, original_name)
            
            # Copy the file
            shutil.copy2(file_path, destination_path)

    print(f"Dataset organized. {len(image_files)} images processed.")
    return gesture_to_idx

# Usage
source_folder = 'organized_dataset_skele'
destination_folder = 'skele_org'
test_split = 0.2  # 20% for test set
gesture_to_idx = organize_dataset(source_folder, destination_folder, test_split)

print(f"Number of classes: {len(gesture_to_idx)}")
print(f"Class mapping: {gesture_to_idx}")
