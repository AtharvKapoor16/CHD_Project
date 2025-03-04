import os
import pandas as pd
import torch
from monai.data import Dataset, DataLoader
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, Resize, ToTensor
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim
from monai.networks.nets import DenseNet264
from tqdm import tqdm

# Constants
CSV_FILE = "labeled_frames.csv"
IMAGE_SIZE = (224, 224)  # Resize all images to this size (adjust as needed)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
MODEL_PATH = "echocardiogram_model.pth"

# Load CSV with paths and labels
df = pd.read_csv(CSV_FILE)

# Assuming the CSV has columns 'file_path' and 'label'
image_paths = df['file_path'].tolist()
labels = df['label'].tolist()

# Check for invalid file paths
invalid_paths = [path for path in image_paths if not os.path.exists(path)]
if invalid_paths:
    print("Invalid image paths found:", invalid_paths)

# Define the transformation pipeline for the images
transforms = Compose([
    EnsureChannelFirst(),
    Resize(IMAGE_SIZE),
    ScaleIntensity(),
    ToTensor()
])

# Prepare the dataset
class EchocardiogramDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Check if the image path exists
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            return torch.zeros((1, 224, 224)), label  # Return default image (modify as needed)

        # Load the image
        image = LoadImage(image_only=True)(image_path)
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Split dataset into training and validation sets
dataset = EchocardiogramDataset(image_paths, labels, transform=transforms)
train_size = int(0.8 * len(dataset))  # 80% training data
val_size = len(dataset) - train_size  # 20% validation data
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoader for batching
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define the model (using DenseNet264 with an additional fully connected layer)
class ModifiedDenseNet264(nn.Module):
    def __init__(self):
        super(ModifiedDenseNet264, self).__init__()
        self.base_model = DenseNet264(spatial_dims=2, in_channels=1, out_channels=512)  # Intermediate features
        self.fc = nn.Linear(512, 1)  # Additional fully connected layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc(x)
        return self.sigmoid(x)  # Output between 0 and 1

# Instantiate the model, loss function, and optimizer
model = ModifiedDenseNet264()
criterion = nn.BCELoss()  # Binary Cross Entropy loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Early stopping parameters
patience = 5  # Number of epochs to wait for improvement
best_val_loss = float('inf')
epochs_without_improvement = 0

# Training loop with early stopping and progress bar
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Training loop with progress bar
    for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}', leave=False):
        inputs, targets = inputs.to(device), targets.float().to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Accuracy calculation
        predicted = (outputs.squeeze() > 0.5).float()
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    # Accuracy on training set
    train_accuracy = correct / total
    print(f"Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader):.4f}, Training Accuracy: {train_accuracy:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.float().to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs.squeeze(), targets).item()
            predicted = (outputs.squeeze() > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    val_accuracy = correct / total
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        # Save model if it improves
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

# Save the final model
torch.save(model.state_dict(), MODEL_PATH)
print(f"Final model saved to {MODEL_PATH}")