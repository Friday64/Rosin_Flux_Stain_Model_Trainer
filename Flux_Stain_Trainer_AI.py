# Standard library imports
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

# Debugging function to print message with variable state
def debug_print(message, variable=None):
    if variable is not None:
        print(f"DEBUG: {message}: {variable}")
    else:
        print(f"DEBUG: {message}")

# Paths to image folders and model
with_flux_folder = "/path/to/With_Flux"
without_flux_folder = "/path/to/Without_Flux"
output_folder = "/path/to/Flux_Models"

# Size to which images will be resized
img_size = (128, 128)

# Define your neural network class
class FluxNet(nn.Module):
    def __init__(self):
        super(FluxNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Data handling in PyTorch
class FluxDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, img_size)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        if self.transform:
            image = self.transform(image)
        label = 1 if self.directory == with_flux_folder else 0
        return image, label

# Training function adapted for PyTorch with CPU usage
def train_model_pytorch(epochs):
    debug_print("Training model with epochs", epochs)

    device = torch.device("cpu")
    model = FluxNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = FluxDataset(with_flux_folder, transform=transform)
    test_dataset = FluxDataset(without_flux_folder, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        debug_print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

    torch.save(model.state_dict(), f"{output_folder}/flux_model.pth")
    debug_print("Training complete, model saved at", f"{output_folder}/flux_model.pth")
    messagebox.showinfo("Training Complete", "Model trained and saved successfully.")

# Function to start training in a separate thread
def start_training():
    epochs = int(epochs_entry.get())
    debug_print("start_training called, starting thread")
    threading.Thread(target=train_model_pytorch, args=(epochs,)).start()

# Initialize Tkinter window
window = tk.Tk()
window.title("Flux Stain Detector")

# UI elements
tk.Label(window, text="Number of Epochs:").pack()
epochs_entry = tk.Entry(window)
epochs_entry.pack()

train_button = tk.Button(window, text="Train Model", command=start_training)
train_button.pack()

# Run the Tkinter event loop
window.mainloop()
