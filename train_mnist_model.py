import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
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

# Download and load MNIST dataset
print("Loading MNIST dataset...")
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Define neural network model
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialize model, loss function, and optimizer
print("Building neural network model...")
model = MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\nModel Architecture:")
print(model)

# Train the model
print("\nTraining model (15 epochs)...")
num_epochs = 15
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

        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    avg_train_loss = train_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

# Evaluate on test set
print("\nEvaluating model on test set...")
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
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Save model weights to JSON for JavaScript
print("\nConverting model to JSON format...")
weights_dict = {}
for name, param in model.named_parameters():
    if 'weight' in name or 'bias' in name:
        weights_dict[name] = param.cpu().detach().numpy().tolist()

# Save model structure and weights
model_data = {
    'type': 'dense_network',
    'input_size': 784,
    'layers': [
        {'type': 'dense', 'units': 128, 'activation': 'relu'},
        {'type': 'dropout', 'rate': 0.2},
        {'type': 'dense', 'units': 64, 'activation': 'relu'},
        {'type': 'dropout', 'rate': 0.2},
        {'type': 'dense', 'units': 32, 'activation': 'relu'},
        {'type': 'dense', 'units': 10, 'activation': 'softmax'}
    ],
    'output_size': 10,
    'weights': weights_dict,
    'test_accuracy': test_accuracy
}

# Save as JSON
with open('model/tfjs_model/model.json', 'w') as f:
    json.dump(model_data, f)

# Also save PyTorch model
torch.save(model.state_dict(), 'model/mnist_model.pth')

print("\nModel training completed!")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print("Model saved to: model/tfjs_model/model.json and model/mnist_model.pth")
print("\nYou can now start the web server with: python run_server.py")
