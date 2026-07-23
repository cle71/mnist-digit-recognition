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

# Data augmentation
transform_train = transforms.Compose([
    transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.85, 1.15)),
    transforms.ToTensor(),
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

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Improved Dense Neural Network
class ImprovedMNISTNet(nn.Module):
    def __init__(self):
        super(ImprovedMNISTNet, self).__init__()

        # Input: 784 neurons
        self.fc1 = nn.Linear(784, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(0.2)

        self.fc5 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)

        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout3(x)

        # Layer 4
        x = self.fc4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.dropout4(x)

        # Output
        x = self.fc5(x)

        return x

# Initialize model
print("Building improved neural network model...")
model = ImprovedMNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=0.00001)

print("\nModel Architecture:")
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training
print("\nTraining model (25 epochs with data augmentation)...")
num_epochs = 25
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

    print(f'Epoch [{epoch+1:2d}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:6.2f}%, Test Acc: {test_accuracy:6.2f}%')

    # Save best model
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), 'model/mnist_improved_model.pth')
        print(f'                          ✓ New best model: {test_accuracy:.2f}%')

    scheduler.step()

print(f"\n{'='*70}")
print(f"Final Test Accuracy: {test_accuracy:.2f}%")
print(f"Best Test Accuracy: {best_accuracy:.2f}%")
print(f"{'='*70}\n")

# Save weights in JSON format for JavaScript
print("Converting model to JSON format for web...")

# Load best model
model.load_state_dict(torch.load('model/mnist_improved_model.pth'))
model.eval()

weights_dict = {}
for name, param in model.named_parameters():
    if 'weight' in name or 'bias' in name:
        weights_dict[name] = param.cpu().detach().numpy().tolist()

# Model metadata
model_data = {
    'type': 'improved_dense_network',
    'input_size': 784,
    'layers': [
        {'type': 'dense', 'units': 512, 'activation': 'relu'},
        {'type': 'batchnorm', 'units': 512},
        {'type': 'dropout', 'rate': 0.3},
        {'type': 'dense', 'units': 256, 'activation': 'relu'},
        {'type': 'batchnorm', 'units': 256},
        {'type': 'dropout', 'rate': 0.3},
        {'type': 'dense', 'units': 128, 'activation': 'relu'},
        {'type': 'batchnorm', 'units': 128},
        {'type': 'dropout', 'rate': 0.2},
        {'type': 'dense', 'units': 64, 'activation': 'relu'},
        {'type': 'batchnorm', 'units': 64},
        {'type': 'dropout', 'rate': 0.2},
        {'type': 'dense', 'units': 10, 'activation': 'softmax'}
    ],
    'output_size': 10,
    'weights': weights_dict,
    'test_accuracy': test_accuracy,
    'best_accuracy': best_accuracy,
    'training_epochs': num_epochs,
    'features': [
        'Batch Normalization',
        'Dropout regularization',
        'Data augmentation',
        'Adam optimizer with weight decay',
        'Cosine annealing learning rate scheduler'
    ]
}

# Save as JSON
with open('model/tfjs_model/model.json', 'w') as f:
    json.dump(model_data, f)

# Also save PyTorch model
torch.save(model.state_dict(), 'model/mnist_improved_model.pth')

print("Model training completed!")
print(f"Final Test Accuracy: {test_accuracy:.2f}%")
print("Model saved to: model/tfjs_model/model.json")
print("                model/mnist_improved_model.pth")
