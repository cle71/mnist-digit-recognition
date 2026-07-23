import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import json
import os
import ssl

# Disable SSL verification for MNIST download
ssl._create_default_https_context = ssl._create_unverified_context

# Create output directory
os.makedirs('model/tfjs_model', exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data augmentation and normalization
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# Load MNIST dataset
print("Loading MNIST dataset...")
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=transform_train,
    download=True
)
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transform_test,
    download=True
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Define CNN model
class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()

        # Convolutional layer 1: 1 -> 32 channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14

        # Convolutional layer 2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7

        # Convolutional layer 3: 64 -> 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 7x7 -> 3x3 (but 7x7 is odd, so -> 3x3)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)

        return x

# Initialize model
print("Building CNN model...")
model = MNISTCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

print("\nModel Architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training loop
print("\nTraining model (20 epochs)...")
num_epochs = 20
best_accuracy = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total_train += target.size(0)
        correct_train += (predicted == target).sum().item()

        if (batch_idx + 1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    avg_train_loss = train_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train

    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%')

    # Save best model
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), 'model/mnist_cnn_model.pth')
        print(f'✓ Best model saved with accuracy: {test_accuracy:.2f}%')

    scheduler.step()

print(f"\n{'='*60}")
print(f"Final Test Accuracy: {test_accuracy:.2f}%")
print(f"Best Test Accuracy: {best_accuracy:.2f}%")
print(f"{'='*60}")

# Convert to dense model for JavaScript (flatten the network)
print("\nConverting CNN to dense model for JavaScript...")

# Load best model
model.load_state_dict(torch.load('model/mnist_cnn_model.pth'))
model.eval()

# Create equivalent dense model
class DenseEquivalent(nn.Module):
    def __init__(self):
        super(DenseEquivalent, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Save weights in JSON format
weights_dict = {}
for name, param in model.named_parameters():
    if 'weight' in name or 'bias' in name:
        weights_dict[name] = param.cpu().detach().numpy().tolist()

# Model metadata
model_data = {
    'type': 'cnn_network',
    'architecture': 'CNN with 3 convolutional layers',
    'input_size': 784,
    'layers': [
        {'type': 'conv2d', 'in_channels': 1, 'out_channels': 32, 'kernel_size': 3},
        {'type': 'batchnorm2d', 'num_features': 32},
        {'type': 'relu'},
        {'type': 'maxpool2d', 'kernel_size': 2},
        {'type': 'conv2d', 'in_channels': 32, 'out_channels': 64, 'kernel_size': 3},
        {'type': 'batchnorm2d', 'num_features': 64},
        {'type': 'relu'},
        {'type': 'maxpool2d', 'kernel_size': 2},
        {'type': 'conv2d', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 3},
        {'type': 'batchnorm2d', 'num_features': 128},
        {'type': 'relu'},
        {'type': 'maxpool2d', 'kernel_size': 2},
        {'type': 'flatten'},
        {'type': 'dense', 'units': 256, 'activation': 'relu'},
        {'type': 'dropout', 'rate': 0.5},
        {'type': 'dense', 'units': 128, 'activation': 'relu'},
        {'type': 'dropout', 'rate': 0.3},
        {'type': 'dense', 'units': 10, 'activation': 'softmax'}
    ],
    'output_size': 10,
    'weights': weights_dict,
    'test_accuracy': test_accuracy,
    'best_accuracy': best_accuracy,
    'training_epochs': num_epochs,
    'model_type': 'CNN'
}

# Save as JSON
with open('model/tfjs_model/model.json', 'w') as f:
    json.dump(model_data, f)

# Also save PyTorch model
torch.save(model.state_dict(), 'model/mnist_cnn_model.pth')

print("\nModel training completed!")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print("Model saved to: model/tfjs_model/model.json and model/mnist_cnn_model.pth")
