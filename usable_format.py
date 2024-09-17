# Set the folder containing all the images


import os
import shutil

# Paths to the original and new dataset directories
original_dataset_path = "archive/leapGestRecog"
new_dataset_path = "organized_dataset"

# Create the new dataset folder if it doesn't exist
if not os.path.exists(new_dataset_path):
    os.makedirs(new_dataset_path)

# Traverse through the original dataset
for subject_folder in os.listdir(original_dataset_path):
    subject_path = os.path.join(original_dataset_path, subject_folder)
    
    if os.path.isdir(subject_path):
        # Loop through each pose folder inside the subject folder
        for pose_folder in os.listdir(subject_path):
            pose_path = os.path.join(subject_path, pose_folder)

            if os.path.isdir(pose_path):
                # Create the corresponding pose folder in the new dataset if it doesn't exist
                new_pose_path = os.path.join(new_dataset_path, pose_folder)
                if not os.path.exists(new_pose_path):
                    os.makedirs(new_pose_path)
                
                # Copy all images from the current pose folder to the new pose folder
                for image_file in os.listdir(pose_path):
                    image_file_path = os.path.join(pose_path, image_file)

                    # Check if it's a file (to avoid copying subdirectories if any)
                    if os.path.isfile(image_file_path):
                        shutil.copy(image_file_path, new_pose_path)
                        # If you want to move instead of copy, use shutil.move()

print("Dataset reorganized by poses successfully!")



