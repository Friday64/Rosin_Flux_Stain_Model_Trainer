import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.regularizers import l2, l1
from keras.models import Sequential
from keras.layers import Dense
import tensorflow_model_optimization as tfmot
from keras.callbacks import LambdaCallback
import tensorflow as tf
import threading

# Global flag to control training
stop_training = False

# Hardcoded folder paths
with_flux_folder = "path/to/with_flux_folder"
without_flux_folder = "path/to/without_flux_folder"
output_folder = "path/to/output_folder"
    
# Function to preprocess and load images
def preprocess_and_load_images(directory_path, img_size):
    image_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path)]
    dataset = []
    for image_file in image_files:
        img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size)
        dataset.append(img)
    return np.array(dataset)


def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Function to convert the model to TFLite
def convert_to_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(f"{output_folder}/flux_detector.tflite", "wb") as f:
        f.write(tflite_model)

# Function to apply pruning to model layers
def apply_pruning_to_layers(model):
    return tfmot.sparsity.keras.prune_low_magnitude(model)

# Function to train the model
def train_model(epochs, Flux_Model):
    train_data = preprocess_and_load_images(with_flux_folder, (28, 28))
    train_labels = np.ones(train_data.shape[0])
    test_data = preprocess_and_load_images(without_flux_folder, (28, 28))
    test_labels = np.zeros(test_data.shape[0])
    data = np.vstack([train_data, test_data])
    labels = np.hstack([train_labels, test_labels])
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    Flux_Model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=64, callbacks=[EarlyStopping(monitor='val_loss', patience=3)], verbose=1)
    Flux_Model.save(f"{output_folder}/flux_model.h5")
    convert_to_tflite(Flux_Model)

# Function to stop training
def stop_training_model():
    global stop_training
    stop_training = True

def start_training_thread():
    global stop_training  # Reset the flag when starting new training
    stop_training = False
    epochs = int(epochs_entry.get())
    Flux_Model = create_model()
    Flux_Model = apply_pruning_to_layers(Flux_Model)
    threading.Thread(target=train_model, args=(epochs, Flux_Model)).start()

def stop_training():
    global stop_training  # Access the global variable
    stop_training = True  # Set it to True to stop training


# Initialize Tkinter window
window = tk.Tk()
window.title("Flux Stain Detector")

# Create UI elements
tk.Label(window, text="Number of Epochs:").pack()
epochs_entry = tk.Entry(window)
epochs_entry.pack()
tk.Button(window, text="Train Model", command=start_training_thread).pack()
tk.Button(window, text="Stop Training", command=stop_training_model).pack()

# Run the Tkinter event loop
window.mainloop()