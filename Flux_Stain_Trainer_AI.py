import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import tensorflow as tf
import tensorflow as tf
import threading

# Global Variables
with_flux_folder = ""
without_flux_folder = ""
output_folder = "output"
stop_training = False

def browse_folder(label_widget, folder_type):
    folder = filedialog.askdirectory(title=f"Select {folder_type} Folder")
    if folder:
        label_widget.config(text=folder)
        global with_flux_folder, without_flux_folder
        if folder_type == "With Flux":
            with_flux_folder = folder
        else:
            without_flux_folder = folder

def browse_output(label_widget):
    global output_folder
    output_folder = filedialog.askdirectory(title="Select Output Folder")
    if output_folder:
        label_widget.config(text=output_folder)

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

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def convert_to_tflite(model):

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(os.path.join(output_folder, "Flux_Stain_Model.tflite"), "wb") as f:
        f.write(tflite_model)
    print("Model converted to TFLite format.")

def train_model(epochs):
    global stop_training
    stop_training = False
    
    model_path = os.path.join(output_folder, 'Flux_Stain_Model.h5')
    model = load_model(model_path) if os.path.exists(model_path) else create_model()

    with_flux_data = preprocess_and_load_images(with_flux_folder, 0)
    without_flux_data = preprocess_and_load_images(without_flux_folder, 1)
    all_data = with_flux_data + without_flux_data

    X = np.array([i[0] for i in all_data])
    y = to_categorical([i[1] for i in all_data])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[early_stop])
    
    model.save(model_path)
    if not stop_training:
        convert_to_tflite(model)
    
    print("Model training complete and saved!")


    model.save(model_path)
    if not stop_training:
        convert_to_tflite(model)
    print("Model training complete and saved!")

def start_training_thread():
    try:
        num_epochs = int(epochs_entry.get())
        if num_epochs < 1:
            print("Please enter a valid number of epochs.")
            return
        print(f"Training started with {num_epochs} epochs.")
        threading.Thread(target=train_model, args=(num_epochs,)).start()
    except ValueError:
        print("Please enter a valid number for epochs.")

def stop_training():
    global stop_training
    stop_training = True

# Tkinter GUI
window = tk.Tk()
window.title("Flux Stain Detector")

folders = [("With Flux", 'with_flux_display'), ("Without Flux", 'without_flux_display')]
for folder_type, label_name in folders:
    tk.Label(window, text=f"{folder_type} Folder:").pack()
    label_name = tk.Label(window, text="Not Selected")
    label_name.pack()
    tk.Button(window, text="Browse", command=lambda l=label_name, f=folder_type: browse_folder(l, f)).pack()

output_display = tk.Label(window, text="Not Selected")
output_display.pack()
tk.Label(window, text="Output Folder:").pack()
tk.Button(window, text="Browse", command=lambda: browse_output(output_display)).pack()

epochs_entry = tk.Entry(window)
epochs_entry.pack()
tk.Label(window, text="Number of Epochs:").pack()

tk.Button(window, text="Start Training", command=start_training_thread).pack()
tk.Button(window, text="Stop Training", command=stop_training).pack()

window.mainloop()
