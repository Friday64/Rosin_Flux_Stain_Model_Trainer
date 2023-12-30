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
import tkinter as tk
from tkinter import messagebox
import logging

device = torch.device("cpu")
print(f"Using device: {device}")


# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants for paths and hyperparameters
WITH_FLUX_FOLDER = "C:/Users/Matthew/Desktop/Programming/Detect_Flux_Project/Flux_Data/With_Flux"  # Update this path
WITHOUT_FLUX_FOLDER = "C:/Users/Matthew/Desktop/Programming/Detect_Flux_Project/Flux_Data/Without_Flux"  # Update this path
OUTPUT_FOLDER = "C:/Users/Matthew/Desktop/Programming/Detect_Flux_Project/Flux_Models"  # Update this path
IMG_SIZE = (256, 256)
LEARNING_RATE = 0.00001
BATCH_SIZE = 64  # Adjust as needed

# Load data paths and labels
all_data = []
all_labels = []

# Load images with flux (label 1)
for filename in os.listdir(WITH_FLUX_FOLDER):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        all_data.append(os.path.join(WITH_FLUX_FOLDER, filename))
        all_labels.append(1)

# Load images without flux (label 0)
for filename in os.listdir(WITHOUT_FLUX_FOLDER):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        all_data.append(os.path.join(WITHOUT_FLUX_FOLDER, filename))
        all_labels.append(0)

# Split the data into training and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(all_data, all_labels, test_size=0.2)

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
        image = cv2.resize(image, IMG_SIZE)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image).float()
        return image, label

# Create datasets and dataloaders
train_dataset = FluxDataset(train_data, train_labels, transform=transforms.ToTensor())
val_dataset = FluxDataset(val_data, val_labels, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Path to save or load the model
model_path = f"{OUTPUT_FOLDER}/flux_model.pth"

# Function to check for the model and train if not present
def check_and_train_model(model_path, train_loader, epochs):
    model = FluxNet().to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info("Model loaded successfully.")
        model = train_model_pytorch(train_loader, model, epochs, device, model_path)
        logging.info("retraining complete")
    else:
        logging.info("No pre-trained model found. Training a new model...")
        model = train_model_pytorch(train_loader, model, epochs, device, model_path)
        
    return model

# Define a function for training
def train_model_pytorch(train_loader, model, epochs, device, model_path):
    print("Training function called.")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    try:
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
            
            # Print training metrics for each epoch
            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}, F1 Score: {epoch_f1:.4f}")

        torch.save(model.state_dict(), model_path)
        print("Training complete, model saved at", model_path)
        messagebox.showinfo("Training Complete", "Model trained and saved successfully.")

    except Exception as e:
        print("Error during training:", str(e))

    return model

if __name__ == "__main__":
    # Tkinter UI setup
    def start_training():
        train_button.config(state=tk.DISABLED)  # Disable the button while training
        epochs = epochs_entry.get()
        if not epochs.isdigit():  # Simple validation to ensure epochs is a number
            messagebox.showerror("Error", "Please enter a valid number of epochs.")
            return

        logging.info(f"Requested training with {epochs} epochs.")
        try:
            # Call the training function directly
            check_and_train_model(model_path, train_loader, int(epochs))
        except Exception as e:
            messagebox.showerror("Training Error", f"An error occurred: {str(e)}")
            logging.error("Training Error:", exc_info=True)

        train_button.config(state=tk.NORMAL) 
    window = tk.Tk()
    window.title("Flux Stain Detector")

    tk.Label(window, text="Number of Epochs:").pack()
    epochs_entry = tk.Entry(window)
    epochs_entry.pack()
    train_button = tk.Button(window, text="Train Model", command=start_training)
    train_button.pack()

    window.mainloop()  # Start the Tkinter event loop
