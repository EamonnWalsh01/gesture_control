import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
print('1')
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch.nn.functional as F
dir="/dir"
class pose_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=3, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(3,3), stride=3, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flat = nn.Flatten()
        self.fc3 = nn.Linear(32 * 12 * 12, 512)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(512, 10)

    
    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        x = self.act2(self.conv2(x))
        x = self.pool2(x)
    
      # Use print() to display the shape
    
        x = self.flat(x)
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output




if __name__ == "__main__":
    print('1')
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.device_count())
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())

    # Paths to the train and test directories
    train_dir = "organized_dataset/train"
    test_dir = "organized_dataset/test"

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the datasets and create DataLoaders
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    print('2')
    batch_size = 32
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    print('3')

    # Model training code
    model = pose_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    n_epochs =10
    for epoch in range(n_epochs):
    # Training loop
        model.train()
        train_pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{n_epochs} [Train]', leave=True)
        for inputs, labels in train_pbar:
            # Forward pass, backward pass, and weight update
            y_pred = model(inputs)
            loss = loss_fn(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Testing loop
    model.eval()
    acc = 0
    count = 0
    test_pbar = tqdm(testloader, desc=f'Epoch {epoch+1}/{n_epochs} [Test]', leave=False)
    with torch.no_grad():
        for inputs, labels in test_pbar:
            y_pred = model(inputs)
            acc += (torch.argmax(y_pred, 1) == labels).float().sum()
            count += len(labels)
            test_pbar.set_postfix({'acc': f'{(acc/count)*100:.2f}%'})

    acc /= count
    print(f"Epoch {epoch+1}/{n_epochs}: model accuracy {acc*100:.2f}%")    
    torch.save(model.state_dict(), "pose_model.pth")


    # Visualization code
    image_tensor, label = train_dataset[7]
    X = image_tensor.unsqueeze(0)
    print(X.shape)

    model.eval()
    with torch.no_grad():
        feature_maps = model.conv1(X)

    fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))
    for i in range(32):
        row, col = i // 8, i % 8
        ax[row][col].imshow(feature_maps[0][i].cpu().numpy())

    plt.tight_layout()
    plt.show()
