import cv2
import os
import torch 
import torch
from conv import CIFAR10Model  # Make sure to import your model class

# Create an instance of your model
model = CIFAR10Model()

# Load the state dictionary
state_dict = torch.load("cifar10model.pth")

# Load the state dict into your model
model.load_state_dict(state_dict)

# Now you can call eval()
model.eval()
# Use os.path.join for cross-platform compatibility
im_path = os.path.join('organized_dataset', 'test','07_ok', 'frame_04_07_0187.png')

# Check if the file exists
if not os.path.exists(im_path):
    print(f"Error: The file {im_path} does not exist.")
    exit()

# Read the image
image = cv2.imread(im_path)


# Check if the image was successfully read
if image is None:
    print(f"Error: Unable to read the image at {im_path}")
    exit()

# Display the image
cv2.imshow('image', image)

# Wait for a key press
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()()
