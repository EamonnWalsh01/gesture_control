import matplotlib.pyplot as plt
import torch
import torchvision.models as models  # Assuming you're using a model from torchvision
import torchvision
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)




# 1. Create an instance of the same model architecture
# Replace 'ResNet18' with whatever model architecture you used
model = models.resnet18(pretrained=False)

# 2. Load the state dictionary
state_dict = torch.load("cifar10model.pth")

# 3. Load the state dictionary into the model
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()
X = torch.tensor([trainset.data[7]], dtype=torch.float32).permute(0,3,1,2)
model.eval()
with torch.no_grad():
    feature_maps = model.conv1(X)
fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))
for i in range(0, 32):
    row, col = i//8, i%8
    ax[row][col].imshow(feature_maps[0][i])
plt.show()   
