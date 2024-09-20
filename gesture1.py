import cv2
import os
import torch 
import torch
from conv import CIFAR10Model  # Make sure to import your model class
import numpy as np
from torchvision import datasets, transforms

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
im_path= "opencv_frame_0.png"
# Check if the file exists
if not os.path.exists(im_path):
    print(f"Error: The file {im_path} does not exist.")
    exit()

# Read the image
image = cv2.imread(im_path)
print(f"Image shape: {image.shape}")

# Preprocess the image
# Preprocess the image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
image = cv2.resize(image, (224, 224))  # Resize to 224x224
image = image.astype(np.float32) / 255.0  # Convert to float and normalize to [0, 1]
image = np.transpose(image, (2, 0, 1))  # Change from (H, W, C) to (C, H, W)

# Convert to tensor and add batch dimension
test_example = torch.from_numpy(image).unsqueeze(0)

# Apply the same normalization as in your training transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_example = normalize(test_example)
print(f"Tensor shape: {test_example.shape}")
# Ensure the model is on the same device as the input
device = next(model.parameters()).device
test_example = test_example.to(device)



with torch.no_grad():  # Disable gradient calculation
    output = model(test_example)

predicted_class = torch.argmax(output).item()
print(f"Predicted class: {predicted_class}")
test_example = torch.tensor([image])  # Convert your data to a tensor
with torch.no_grad():  # Disable gradient calculation
    output = model(test_example)
# Check if the image was successfully read
predicted_class = torch.argmax(output).item()
if image is None:
    print(f"Error: Unable to read the image at {im_path}")
    exit()

