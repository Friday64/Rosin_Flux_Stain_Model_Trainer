# Standard library imports
import logging
import os
import tkinter as tk
from tkinter import messagebox

# Related third-party imports
from keras import layers, models, optimizers
from keras.applications import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
import logging
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants for paths and hyperparameters
WITH_FLUX_FOLDER = "C:/Users/Matthew/Desktop/Programming/Detect_Flux_Project/Flux_Data/With_Flux"
WITHOUT_FLUX_FOLDER = "C:/Users/Matthew/Desktop/Programming/Detect_Flux_Project/Flux_Data/Without_Flux"
MODEL_PATH = "C:/Users/Matthew/Desktop/Programming/Detect_Flux_Project/Flux_Models/flux_model_tf"
IMG_SIZE = (256, 256)
LEARNING_RATE = 0.1
BATCH_SIZE = 32

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Load data paths and labels
def load_data_paths():
    all_data = []
    all_labels = []
    for folder, label in [(WITH_FLUX_FOLDER, 1), (WITHOUT_FLUX_FOLDER, 0)]:
        for filename in os.listdir(folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                all_data.append(os.path.join(folder, filename))
                all_labels.append(label)
    return all_data, all_labels

# Transfer Learning model creation function
def create_model():
    base_model = MobileNetV2(input_shape=(256, 256, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()  # Print the model structure

    return model

# Data augmentation
def get_data_augmentation():
    return ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

# Preprocess and load data using tf.data
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

def load_dataset(data_paths, labels, batch_size):
    path_ds = tf.data.Dataset.from_tensor_slices(data_paths)
    image_ds = path_ds.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))
    dataset = tf.data.Dataset.zip((image_ds, label_ds))
    return dataset.shuffle(len(data_paths)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# Learning Rate Scheduler
def get_optimizer():
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=10000,
        decay_rate=0.9)
    return optimizers.Adam(learning_rate=lr_schedule)

# Function to train the model
def train_model(model, train_ds, val_ds, epochs):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    return history

# Function to save the model
def save_model(model, model_path):
    model.save(model_path)

# Function to load or create model
def load_or_create_model(model_path):
    if os.path.exists(model_path):
        print("Loading existing model.")
        return models.load_model(model_path)
    else:
        print("Creating new model.")
        return create_model()

# Function to convert the model to TensorFlow Lite
def convert_to_tflite(model_path, tflite_model_path, quantize=False):
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f'Model converted to TensorFlow Lite and saved to {tflite_model_path}')

# Tkinter GUI setup for training
def start_training(train_ds, val_ds, val_labels):
    train_button.config(state=tk.DISABLED)
    epochs = epochs_entry.get()
    if not epochs.isdigit():
        messagebox.showerror("Error", "Please enter a valid number of epochs.")
        train_button.config(state=tk.NORMAL)
        return

    logging.info(f"Requested training with {epochs} epochs.")
    try:
        model = load_or_create_model(MODEL_PATH)
        history = train_model(model, train_ds, val_ds, int(epochs))
        save_model(model, MODEL_PATH)

        # Convert and save the TensorFlow Lite model
        tflite_model_path = MODEL_PATH + "_tflite"
        convert_to_tflite(MODEL_PATH, tflite_model_path, quantize=True)

        # Model Evaluation
        predictions = model.predict(val_ds)
        predicted_classes = np.argmax(predictions, axis=1)
        print(classification_report(val_labels, predicted_classes, zero_division=1))

        messagebox.showinfo("Training Complete", "Model trained and saved successfully.")
    except Exception as e:
        messagebox.showerror("Training Error", f"An error occurred: {str(e)}")
        logging.error("Training Error:", exc_info=True)

    train_button.config(state=tk.NORMAL)

# Main function
def main():
    all_data, all_labels = load_data_paths()
    train_data, val_data, train_labels, val_labels = train_test_split(all_data, all_labels, test_size=0.2)
    global train_ds, val_ds
    train_ds = load_dataset(train_data, train_labels, BATCH_SIZE)
    val_ds = load_dataset(val_data, val_labels, BATCH_SIZE)

    window = tk.Tk()
    window.title("Flux Stain Detector")

    tk.Label(window, text="Number of Epochs:").pack()
    global epochs_entry
    epochs_entry = tk.Entry(window)
    epochs_entry.pack()
    global train_button
    train_button = tk.Button(window, text="Train Model", command=lambda: start_training(train_ds, val_ds, val_labels))
    train_button.pack()

    window.mainloop()

if __name__ == "__main__":
    main()
