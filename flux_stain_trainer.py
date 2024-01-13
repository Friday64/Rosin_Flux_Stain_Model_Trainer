import os
import cv2
import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tkinter as tk
from tkinter import messagebox
import logging

# Ensure TensorFlow is using GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("TensorFlow GPU device not found. Ensure tensorflow-gpu is installed.")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants for paths and hyperparameters
WITH_FLUX_FOLDER = "C:/Users/Matthew/Desktop/Programming/Detect_Flux_Project/Flux_Data/With_Flux"
WITHOUT_FLUX_FOLDER = "C:/Users/Matthew/Desktop/Programming/Detect_Flux_Project/Flux_Data/Without_Flux"
OUTPUT_FOLDER = "C:/Users/Matthew/Desktop/Programming/Detect_Flux_Project/Flux_Models"
MODEL_PATH = f"{OUTPUT_FOLDER}/flux_model_tf"
IMG_SIZE = (256, 256)
LEARNING_RATE = 0.001
BATCH_SIZE = 64

# Load data paths and labels
all_data = []
all_labels = []

for filename in os.listdir(WITH_FLUX_FOLDER):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        all_data.append(os.path.join(WITH_FLUX_FOLDER, filename))
        all_labels.append(1)

for filename in os.listdir(WITHOUT_FLUX_FOLDER):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        all_data.append(os.path.join(WITHOUT_FLUX_FOLDER, filename))
        all_labels.append(0)

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

# Check if model exists, load it, otherwise create a new one
def load_or_create_model(model_path):
    if os.path.exists(model_path):
        print("Loading existing model.")
        return models.load_model(model_path)
    else:
        print("Creating new model.")
        return create_model()

model = load_or_create_model(MODEL_PATH)
model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Preprocess and load data
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, IMG_SIZE)
    image = image / 255.0
    image = np.expand_dims(image, axis=-1)
    return image

def load_data(data_paths, labels):
    images = np.array([preprocess_image(path) for path in data_paths])
    labels = np.array(labels)
    return images, labels

train_images, train_labels = load_data(train_data, train_labels)
val_images, val_labels = load_data(val_data, val_labels)

# Image Augmentation
augmentation = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Function to train the model with augmentation
def train_model_with_augmentation(model, train_images, train_labels, val_images, val_labels, epochs, batch_size):
    history = model.fit(
        augmentation.flow(train_images, train_labels, batch_size=batch_size),
        validation_data=(val_images, val_labels),
        steps_per_epoch=len(train_images) // batch_size,
        epochs=epochs
    )
    return history

# Save the model
def save_model(model, model_path):
    model.save(model_path)

# Tkinter UI setup for training
def start_training():
    train_button.config(state=tk.DISABLED)
    epochs = epochs_entry.get()
    if not epochs.isdigit():
        messagebox.showerror("Error", "Please enter a valid number of epochs.")
        return

    logging.info(f"Requested training with {epochs} epochs.")
    try:
        history = train_model_with_augmentation(model, train_images, train_labels, val_images, val_labels, int(epochs), BATCH_SIZE)
        save_model(model, MODEL_PATH)

        # Model Evaluation
        predictions = model.predict(val_images)
        predicted_classes = np.argmax(predictions, axis=1)
        print(classification_report(val_labels, predicted_classes))

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
