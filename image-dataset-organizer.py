import os
import shutil
from pathlib import Path

def get_label_from_filename(filename):
    # Strip the file extension and any leading numbers/underscores
    label = os.path.splitext(filename)[0]
    label = ''.join([i for i in label if not i.isdigit() and i != '_'])
    return label

def organize_dataset(source_folder, destination_folder):
    image_files = []
    labels = set()

    # Create destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Walk through all subdirectories
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(root, file)
                label = get_label_from_filename(file)
                image_files.append((full_path, label))
                labels.add(label)

    # Create a mapping of labels to integers
    label_to_idx = {label: idx for idx, label in enumerate(sorted(labels))}

    # Copy files to destination folder
    for file_path, label in image_files:
        # Create a new filename with the format: label_index_originalname.ext
        original_name = Path(file_path).name
        new_name = f"{label}_{label_to_idx[label]:03d}_{original_name}"
        destination_path = os.path.join(destination_folder, new_name)
        
        # Copy the file
        shutil.copy2(file_path, destination_path)

    print(f"Dataset organized. {len(image_files)} images processed.")
    return label_to_idx

# Usage
source_folder = 'leapGestRecog'
destination_folder = 'organized_dataset'
label_to_idx = organize_dataset(source_folder, destination_folder)

print(f"Number of classes: {len(label_to_idx)}")
print(f"Class mapping: {label_to_idx}")
