# Standard library imports
import os
import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import threading

# Paths to image folders and model
with_flux_folder = "C:/Users/Matthew/Desktop/Programming/Detect_Flux_Project/Flux_Data\With_Flux"
without_flux_folder = "C:/Users/Matthew/Desktop/Programming/Detect_Flux_Project/Flux_Data/Without_Flux"
output_folder = "C:/Users/Matthew/Desktop/Programming/Detect_Flux_Project/Flux_Models"

# Size to which images will be resized
img_size = (128, 128)

# Function to preprocess and load images
def preprocess_and_load_images(directory_path, img_size):
    image_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    dataset = np.zeros((len(image_files), img_size[0], img_size[1]), dtype=np.float32)
    for idx, image_file in enumerate(image_files):
        try:
            img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, img_size)
                img = img / 255.0
                dataset[idx] = img
            else:
                print(f"Warning: Image {image_file} could not be loaded and will be skipped.")
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
    print(f"Finished loading and preprocessing {len(dataset)} images from {directory_path}")
    return dataset

# Function to create the machine learning model
def create_model(input_shape=(128, 128, 1), num_classes=2):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train the model in a separate thread
def train_model(epochs):
    try:
        train_data = preprocess_and_load_images(with_flux_folder, img_size)
        train_labels = np.ones(train_data.shape[0])
        test_data = preprocess_and_load_images(without_flux_folder, img_size)
        test_labels = np.zeros(test_data.shape[0])
        data = np.vstack([train_data, test_data])
        labels = np.hstack([train_labels, test_labels])
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        model_file_path = f"{output_folder}/flux_model.h5"
        if os.path.exists(model_file_path):
            model = load_model(model_file_path)
        else:
            model = create_model(input_shape=(128, 128, 1), num_classes=2)
        callbacks_list = [EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)]

        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=64, callbacks=callbacks_list, verbose=1)

        model.save(f"{output_folder}/flux_model.h5")
        messagebox.showinfo("Training Complete", "Model trained and saved successfully.")
    except Exception as e:
        messagebox.showerror("Training Error", f"An error occurred during training: {e}")

# Function to start training in a separate thread
def start_training():
    epochs = int(epochs_entry.get())
    threading.Thread(target=train_model, args=(epochs,)).start()

# Initialize Tkinter window
window = tk.Tk()
window.title("Flux Stain Detector")

# Create UI elements
tk.Label(window, text="Number of Epochs:").pack()
epochs_entry = tk.Entry(window)
epochs_entry.pack()

train_button = tk.Button(window, text="Train Model", command=start_training)
train_button.pack()

# Run the Tkinter event loop
window.mainloop()
