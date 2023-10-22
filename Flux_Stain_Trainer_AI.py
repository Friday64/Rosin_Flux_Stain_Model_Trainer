# Simplified Imports
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

# Global Variables
with_flux_folder = ""
without_flux_folder = ""
output_folder = "output"

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
    return [(cv2.resize(cv2.imread(os.path.join(folder, f)), (100, 100)) / 255.0, label) 
            for f in os.listdir(folder) if f.endswith(".jpg")]

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(epochs):
    model_path = 'Flux_Stain_Model.h5'
    model = load_model(model_path) if os.path.exists(model_path) else create_model()
    
    with_flux_data = preprocess_and_load_images(with_flux_folder, 1)
    without_flux_data = preprocess_and_load_images(without_flux_folder, 0)
    
    all_images, all_labels = zip(*(with_flux_data + without_flux_data))
    X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2)
    
    y_train, y_test = to_categorical(y_train, 2), to_categorical(y_test, 2)
    
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2,
                                 height_shift_range=0.2, horizontal_flip=True)
    datagen.fit(np.array(X_train))
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    
    model.fit(datagen.flow(np.array(X_train), np.array(y_train), batch_size=32),
              epochs=epochs, validation_data=(np.array(X_test), np.array(y_test)),
              callbacks=[early_stopping])
    
    model.save(os.path.join(output_folder, model_path))
    print("Model Saved")

def start_training():
    num_epochs = int(epochs_entry.get())
    train_model(num_epochs)

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
tk.Button(window, text="Start Training", command=start_training).pack()

window.mainloop()
