
import os
import shutil
import random

# Set random seed for reproducibility
random.seed(42)

# Paths to your new dataset with poses
dataset_path = "organized_dataset"
train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")

# Ratio for train/test split
train_ratio = 0.8

# Create train and test directories if they don't exist
for path in [train_path, test_path]:
    if not os.path.exists(path):
        os.makedirs(path)

# Go through each pose folder
for pose_folder in os.listdir(dataset_path):
    pose_path = os.path.join(dataset_path, pose_folder)

    if os.path.isdir(pose_path) and pose_folder not in ["train", "test"]:
        # List all images in the current pose folder
        images = [f for f in os.listdir(pose_path) if os.path.isfile(os.path.join(pose_path, f))]

        # Shuffle the images randomly
        random.shuffle(images)

        # Split into training and testing sets
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        # Create subdirectories for this pose in train/ and test/ if they don't exist
        train_pose_path = os.path.join(train_path, pose_folder)
        test_pose_path = os.path.join(test_path, pose_folder)

        for dir_path in [train_pose_path, test_pose_path]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        # Move the training images to the train folder
        for image in train_images:
            shutil.move(os.path.join(pose_path, image), os.path.join(train_pose_path, image))

        # Move the testing images to the test folder
        for image in test_images:
            shutil.move(os.path.join(pose_path, image), os.path.join(test_pose_path, image))

# After moving, clean up the original pose directories
for pose_folder in os.listdir(dataset_path):
    pose_path = os.path.join(dataset_path, pose_folder)
    if os.path.isdir(pose_path) and pose_folder not in ["train", "test"]:
        shutil.rmtree(pose_path)

print("Dataset successfully split into train and test sets!")
