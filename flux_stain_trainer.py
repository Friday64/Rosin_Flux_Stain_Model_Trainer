import os
import cv2
import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tkinter as tk
from tkinter import messagebox
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants for paths and hyperparameters
WITH_FLUX_FOLDER = "C:/Users/Matthew/Desktop/Programming/Detect_Flux_Project/Flux_Data/With_Flux"
WITHOUT_FLUX_FOLDER = "C:/Users/Matthew/Desktop/Programming/Detect_Flux_Project/Flux_Data/Without_Flux"
OUTPUT_FOLDER = "C:/Users/Matthew/Desktop/Programming/Detect_Flux_Project/Flux_Models"
IMG_SIZE = (256, 256)
LEARNING_RATE = 0.0001
BATCH_SIZE = 32

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

# Neural network class using Keras
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(256, 256, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])
    return model

# Preprocess and load data
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, IMG_SIZE)
    image = image / 255.0
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    return image

def load_data(data_paths, labels):
    images = np.array([preprocess_image(path) for path in data_paths])
    labels = np.array(labels)
    return images, labels

train_images, train_labels = load_data(train_data, train_labels)
val_images, val_labels = load_data(val_data, val_labels)

# Create and compile the model
model = create_model()
model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Function to train the model
def train_model(model, train_images, train_labels, val_images, val_labels, epochs):
    history = model.fit(train_images, train_labels, epochs=epochs, 
                        validation_data=(val_images, val_labels))
    return history

# Save the model
def save_model(model, model_path):
    model.save(model_path)

# Tkinter UI setup
def start_training():
    train_button.config(state=tk.DISABLED)
    epochs = epochs_entry.get()
    if not epochs.isdigit():
        messagebox.showerror("Error", "Please enter a valid number of epochs.")
        return

    logging.info(f"Requested training with {epochs} epochs.")
    try:
        history = train_model(model, train_images, train_labels, val_images, val_labels, int(epochs))
        save_model(model, f"{OUTPUT_FOLDER}/flux_model_tf")
        messagebox.showinfo("Training Complete", "Model trained and saved successfully.")
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

window.mainloop()
