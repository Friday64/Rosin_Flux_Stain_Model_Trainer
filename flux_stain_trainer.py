import os
import tensorflow as tf
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models, optimizers
from keras.mixed_precision import set_global_policy
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tkinter as tk
from tkinter import messagebox
import logging
import numpy as np

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

# Enable mixed precision training
set_global_policy('mixed_float16')

# Load data paths and labels
all_data = []
all_labels = []

for folder, label in [(WITH_FLUX_FOLDER, 1), (WITHOUT_FLUX_FOLDER, 0)]:
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            all_data.append(os.path.join(folder, filename))
            all_labels.append(label)

# Split data into training and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(all_data, all_labels, test_size=0.2)

# Neural network class using Keras
def create_model():
    model = models.Sequential([
        layers.SeparableConv2D(32, (3, 3), padding='same', activation='relu', input_shape=(256, 256, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.SeparableConv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
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

# Preprocess and load data using tf.data
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, IMG_SIZE)
    image /= 255.0
    return image

def load_data(data_paths, labels):
    path_ds = tf.data.Dataset.from_tensor_slices(data_paths)
    image_ds = path_ds.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))
    return tf.data.Dataset.zip((image_ds, label_ds))

train_ds = load_data(train_data, train_labels).batch(BATCH_SIZE)
val_ds = load_data(val_data, val_labels).batch(BATCH_SIZE)

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

# Learning Rate Scheduler
lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=LEARNING_RATE,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = optimizers.Adam(learning_rate=lr_schedule)

# Function to train the model with augmentation
def train_model_with_augmentation(model, train_ds, val_ds, epochs):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
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
        history = train_model_with_augmentation(model, train_ds, val_ds, int(epochs))
        save_model(model, MODEL_PATH)

        # Model Evaluation
        predictions = model.predict(val_ds)
        predicted_classes = np.argmax(predictions, axis=1)

        print(classification_report(val_labels, predicted_classes, zero_division=1))

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
