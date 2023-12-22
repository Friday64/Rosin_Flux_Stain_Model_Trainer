# Standard library imports
import multiprocessing
import os
import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import threading
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Debugging function
def debug_print(message, variable=None):
    if variable is not None:
        print(f"DEBUG: {message}: {variable}")
    else:
        print(f"DEBUG: {message}")

# Paths to image folders and model
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
        all_labels.append(1)  # Label 1 for with flux

# Load images without flux (label 0)
for filename in os.listdir(without_flux_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        all_data.append(os.path.join(without_flux_folder, filename))
        all_labels.append(0)  # Label 0 for without flux

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
        self.fc1 = nn.Linear(64 * 64 * 64, 128)  # Adjusted size
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

# Training function
def train_model_pytorch(epochs):
    # Use GPU if available, else use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FluxNet().to(device)
    criterion = nn.CrossEntropyLoss()  # Adjust for class weights if necessary
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
        debug_print(f"Epoch {epoch+1}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}, Precision: {epoch_precision}, Recall: {epoch_recall}, F1 Score: {epoch_f1}")

    torch.save(model.state_dict(), f"{output_folder}/flux_model.pth")
    debug_print("Training complete, model saved at", f"{output_folder}/flux_model.pth")
    messagebox.showinfo("Training Complete", "Model trained and saved successfully.")

# Corrected start_training function
def start_training():
    train_button.config(state=tk.DISABLED)  # Disable the button during training
    epochs = int(epochs_entry.get())
    train_model_pytorch(epochs)
    train_button.config(state=tk.NORMAL)  # Re-enable the button after training

# Tkinter UI setup
window = tk.Tk()
window.title("Flux Stain Detector")

# Function to handle window closing
def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        window.destroy()

# Bind the function to the window's close event
window.protocol("WM_DELETE_WINDOW", on_closing)

tk.Label(window, text="Number of Epochs:").pack()
epochs_entry = tk.Entry(window)
epochs_entry.pack()
train_button = tk.Button(window, text="Train Model", command=start_training)
train_button.pack()

# Running the Tkinter main loop
window.mainloop()