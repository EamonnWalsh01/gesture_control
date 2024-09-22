import cv2
import os
import torch 
from conv import pose_model  # Make sure to import your model class
import numpy as np
from torchvision import transforms


cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0
def get_class_names(dataset_path):
    train_path = os.path.join(dataset_path, 'train')
    class_names = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
    return class_names

dataset_path = 'organized_dataset'  
gesture_names = get_class_names(dataset_path)

model = pose_model()

state_dict = torch.load("pose_model.pth")

model.load_state_dict(state_dict)

model.eval()




prediction_text = ""
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
   

   
    display_frame = cv2.resize(frame, (448,448))

    cv2.putText(display_frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        resized_frame = cv2.resize(frame, (224, 224))

        image = cv2.cvtColor(resized_frame , cv2.COLOR_BGR2RGB) 
        image = image.astype(np.float32) / 255.0  # Convert to float and normalize to [0, 1]
        image = np.transpose(image, (2, 0, 1))  # Change from (H, W, C) to (C, H, W)
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
# Get the name of the predicted gesture
                
# Predict the class and get the name of the predicted gesture
        predicted_class = torch.argmax(output).item()

# Initialize prediction_text

# Get the name of the predicted gesture
        if 0 <= predicted_class < len(gesture_names):
            predicted_gesture = gesture_names[predicted_class]
            prediction_text = f"Predicted gesture: {predicted_gesture}"
            print(prediction_text)
        else:
            prediction_text = f"Error: Predicted class {predicted_class} is out of range."

# Display the prediction text
        cv2.putText(display_frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("test", display_frame)
cam.release()

cv2.destroyAllWindows()

