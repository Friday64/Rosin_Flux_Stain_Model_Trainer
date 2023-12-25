import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tkinter import messagebox
from flux_stain_trainer_ai_main import epochs


# Define your dataset paths and output folder
with_flux_folder = "C:/Users/Matthew/Desktop/Programming/Detect_Flux_Project/Flux_Data/With_Flux"
without_flux_folder = "C:/Users/Matthew/Desktop/Programming/Detect_Flux_Project/Flux_Data/Without_Flux"
output_folder = "C:/Users/Matthew/Desktop/Programming/Detect_Flux_Project/Flux_Models"

# Load data paths and labels
all_data = []
all_labels = []

# Load images with flux (label 1)
for filename in os.listdir(with_flux_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        all_data.append(os.path.join(with_flux_folder, filename))
        all_labels.append(1)

# Load images without flux (label 0)
for filename in os.listdir(without_flux_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        all_data.append(os.path.join(without_flux_folder, filename))
        all_labels.append(0)

# Splitting the data into training and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(all_data, all_labels, test_size=0.2)

# Size to which images will be resized
img_size = (256, 256)

# Neural network class
class FluxNet(nn.Module):
    def __init__(self):
        super(FluxNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Dataset class
class FluxDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx]
        label = self.labels[idx]
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, img_size)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image).float()
        return image, label

# Create datasets and dataloaders
train_dataset = FluxDataset(train_data, train_labels, transform=transforms.ToTensor())
val_dataset = FluxDataset(val_data, val_labels, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=1)

# Path to save or load the model
model_path = f"{output_folder}/flux_model.pth"

# Function to check for the model and train if not present
def check_and_train_model(model_path, train_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FluxNet().to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully.")
    else:
        print("No pre-trained model found. Training a new model...")
        model = train_model_pytorch(train_loader, model, epochs, device)  # Training the model
        
    return model

# Modify your training function to be used within check_and_train_model
def train_model_pytorch(train_loader, model, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_predictions = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images.float())
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = accuracy_score(all_labels, all_predictions)
        epoch_precision = precision_score(all_labels, all_predictions, average='binary')
        epoch_recall = recall_score(all_labels, all_predictions, average='binary')
        epoch_f1 = f1_score(all_labels, all_predictions, average='binary')
        print(f"Epoch {epoch+1}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}, Precision: {epoch_precision}, Recall: {epoch_recall}, F1 Score: {epoch_f1}")

    torch.save(model.state_dict(), model_path)
    print("Training complete, model saved at", model_path)
    messagebox.showinfo("Training Complete", "Model trained and saved successfully.")

    return model

if __name__ == "__main__":
    # Parse command-line arguments for epochs
    parser = argparse.ArgumentParser(description='Train Flux Detector Model')
    parser.add_argument('epochs', type=int, help='Number of epochs for training')
    args = parser.parse_args()

    # Use the epochs from the command line
    epochs = args.epochs

    # Rest of your main code
    model = check_and_train_model(model_path, train_loader, epochs)
    # ...[rest of your code]..