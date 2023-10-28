# Import libraries
import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from tkinter import Tk, Label, Button, filedialog, Entry

# Global Variables
with_flux_folder = ""
without_flux_folder = ""
output_folder = "output"

# Function to browse folder
def browse_folder(label_widget, folder_type):
    folder = filedialog.askdirectory(title=f"Select {folder_type} Folder")
    if folder:
        label_widget.config(text=folder)
        global with_flux_folder, without_flux_folder
        if folder_type == "With Flux":
            with_flux_folder = folder
        else:
            without_flux_folder = folder

# Function to browse output folder
def browse_output(label_widget):
    global output_folder
    output_folder = filedialog.askdirectory(title="Select Output Folder")
    if output_folder:
        label_widget.config(text=output_folder)

# Function to preprocess and load images
def preprocess_and_load_images(folder, label):
    if not os.path.exists(folder):
        print(f"Folder {folder} does not exist.")
        return []

    processed_images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(folder, filename))
            img_resized = cv2.resize(img, (100, 100))
            img_normalized = img_resized / 255.0
            processed_images.append((img_normalized, label))
            
    return processed_images

# Function to create model
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train model
def train_model(epochs):
    with_flux_data = preprocess_and_load_images(with_flux_folder, 1)
    without_flux_data = preprocess_and_load_images(without_flux_folder, 0)
    all_data = with_flux_data + without_flux_data
    all_images, all_labels = zip(*all_data)
    all_images = np.array(all_images)
    all_labels = np.array(all_labels)
    
    X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2)
    y_train, y_test = to_categorical(y_train, 2), to_categorical(y_test, 2)
    
    model_path = 'Flux_Stain_Model'
    model = load_model(model_path) if os.path.exists(model_path) else create_model()

    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
    datagen.fit(X_train)

    early_stopping = EarlyStopping(monitor='val_loss', patience=50)
    model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=epochs, validation_data=(X_test, y_test), callbacks=[early_stopping])
    model.save(os.path.join(output_folder, model_path))

# Function to convert to TF Lite
def convert_to_tflite():
    model = load_model(os.path.join(output_folder, "Flux_Stain_Model"))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(os.path.join(output_folder, "Flux_Stain_Model.tflite"), "wb") as f:
        f.write(tflite_model)

# Tkinter GUI
window = Tk()
window.title("Flux Stain Detector")

# Folders for images with flux and without flux
folders = [("With Flux", 'with_flux_display'), ("Without Flux", 'without_flux_display')]
for folder_type, label_name in folders:
    Label(window, text=f"{folder_type} Folder:").pack()
    label_name = Label(window, text="Not Selected")
    label_name.pack()
    Button(window, text="Browse", command=lambda l=label_name, f=folder_type: browse_folder(l, f)).pack()

# Output Folder
output_display = Label(window, text="Not Selected")
output_display.pack()
Label(window, text="Output Folder:").pack()
Button(window, text="Browse", command=lambda: browse_output(output_display)).pack()

# Number of Epochs
epochs_entry = Entry(window)
epochs_entry.pack()
Label(window, text="Number of Epochs:").pack()

# Button to Start Training
Button(window, text="Start Training", command=lambda: train_model(int(epochs_entry.get()))).pack()

# Button to Convert to TFLite
Button(window, text="Convert to TF Lite", command=convert_to_tflite).pack()

window.mainloop()
