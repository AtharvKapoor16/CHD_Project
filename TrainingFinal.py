import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm  # For progress bar
from monai.networks.nets import DenseNet264
from sklearn.model_selection import train_test_split

# Define constants
IMAGE_SIZE = (224, 224)
DATA_DIR = "VSD_Frames"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
MODEL_PATH = "chd_vs_vsd_densenet264.pth"

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),  # Ensure single-channel input
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1,1]
])

# Function to load and preprocess images
def load_and_preprocess_image(image_path):
    """Loads an image, converts it to grayscale, resizes, and normalizes it."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return None
    return transform(img)

# Custom PyTorch Dataset
class EchocardiogramDataset(Dataset):
    def __init__(self, data_dir):
        self.image_paths = []
        self.labels = []
        self.classes = ["without_chd", "VSD_Frames"]

        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(data_dir, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: {class_path} does not exist!")
                continue

            for video_folder in os.listdir(class_path):
                video_path = os.path.join(class_path, video_folder)
                if os.path.isdir(video_path):
                    for img_name in os.listdir(video_path):
                        img_path = os.path.join(video_path, img_name)
                        img = load_and_preprocess_image(img_path)
                        if img is not None:
                            self.image_paths.append(img)
                            self.labels.append(class_idx)

        if not self.image_paths:
            raise ValueError("No valid images found! Check dataset paths.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.image_paths[idx], self.labels[idx]

# Load dataset
print("Loading dataset...")
dataset = EchocardiogramDataset(DATA_DIR)
X_train, X_test, y_train, y_test = train_test_split(dataset.image_paths, dataset.labels, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train, X_test = torch.stack(X_train), torch.stack(X_test)
y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

# Create DataLoaders
train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=BATCH_SIZE, shuffle=False)

# Define MONAI DenseNet264 model
class CHDClassifier(nn.Module):
    def __init__(self):
        super(CHDClassifier, self).__init__()
        self.model = DenseNet264(spatial_dims=2, in_channels=1, out_channels=2)  # 2 classes (without_chd, VSD)

    def forward(self, x):
        return self.model(x)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CHDClassifier().to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop with tqdm progress bar
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

    # Save model checkpoint
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved as {MODEL_PATH}")

# Evaluate the model
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy:.4f}")